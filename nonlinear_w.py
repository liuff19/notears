from cProfile import run
from sympy import re
from notears.locally_connected import LocallyConnected
from notears.trace_expm import trace_expm
import torch
import torch.nn as nn
from notears.lbfgsb_scipy import LBFGSBScipy
import numpy as np
import tqdm as tqdm
from notears.loss_func import *
import random
import time
import igraph as ig
import notears.utils as ut
from notears.loss_func import *
import torch.utils.data as data
from adaptive_model.adapModel import adaptiveMLP
from adaptive_model.adapModel import adap_reweight_step

from runhelps.runhelper import config_parser
from torch.utils.tensorboard import SummaryWriter
from sachs_data.load_sachs import *
import torch.optim as optim
import torch.nn.functional as F
COUNT = 0

IF_figure = 0

parser = config_parser()
args = parser.parse_args()
print(args)
run_mode = args.run_mode # 0: reweight + batch ; 1: baseline; 2: batch + baseline
class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2    
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)  
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers) 
        layers = []
        # TODO: add the discriminator function for wasserstein loss
        layers.append(LocallyConnected(d, d, 1, bias=bias))
        self.fc3 = nn.ModuleList(layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1] 
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d] 
        return x

    def forward__(self, x):  # [n, d] -> [n, d]
        W = 1 - torch.eye (x.shape [1])
        W = W.reshape (1, W.shape [0], -1).expand (x.shape [0], -1, -1) 
        x = x.reshape (x.shape [0], 1, -1).expand (-1, x.shape [1], -1) 
        x = x * W 
        for _, fc in enumerate (self.fc3) :
            if _ != 0 : 
                x = torch.relu (x)  # [n, d, m1]
            #else :
                #x = torch.nn.functional.dropout (x, 0.5)
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x
    
    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i] 
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        # h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        M = torch.eye(d).to(A.device) + A / d  # (Yu et al. 2019)
        E = torch.matrix_power(M, d - 1)
        h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg 

    def l2_reg__(self):
        """Take 2-norm-squared of all parameters in Discriminator """
        reg = 0.
        for fc in self.fc3:
            reg += torch.sum(fc.weight ** 2)
        return reg
    
    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W

def adaptive_loss(output, target, reweight_list):
    R = output-target
    # reweight_matrix = torch.diag(reweight_idx).to(args.device)
    # loss = 0.5 * torch.sum(torch.matmul(reweight_matrix, R))
    loss = 0.5 * torch.sum(torch.mul(reweight_list, R**2))
    return loss


def wasserstein_loss(x_distribution, wx_distribution):
    # TODO: we need to pay attention to the axis of the average
    # TODO: check the correctness of the loss function
    return (torch.mean(x_distribution, axis = 0) - torch.mean(wx_distribution, axis = 0)).sum()
    
def dual_ascent_step(model, X, train_loader, lambda1, lambda2, rho, alpha, h, rho_max, adp_flag, adaptive_model):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    space_optimizer = optim.RMSprop (model.fc3.parameters (), lr = 1e-2, momentum = 0.9)
    # X_torch = torch.from_numpy(X)
    while rho < rho_max:
        def closure__():
            space_optimizer.zero_grad()
            primal_obj = torch.tensor(0.).to(args.device)
            
            for _ , tmp_x in enumerate(train_loader):
                batch_x = tmp_x[0].to(args.device)
                x_distribution = model.forward__(batch_x)
                wx_distribution = model.forward__(model(batch_x))
                primal_obj += -wasserstein_loss(x_distribution, wx_distribution)
            
            l2_reg = 10 * 0.5 * lambda2 * model.l2_reg__ ()
            primal_obj += l2_reg
            primal_obj.backward ()
            return primal_obj
        
        def closure():
            global COUNT
            COUNT += 1
            optimizer.zero_grad()
            X_hat = model(X)
            loss = squared_loss(X_hat, X)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            primal_obj.backward()
            return primal_obj

        def r_closure():
            global COUNT
            COUNT += 1
            optimizer.zero_grad()

            primal_obj = torch.tensor(0.).to(args.device)
            loss = torch.tensor(0.).to(args.device)

            for _ , tmp_x in enumerate(train_loader):
                batch_x = tmp_x[0].to(args.device)
                X_hat = model(batch_x)
            
                if adp_flag == False:
                    reweight_list = torch.ones(batch_x.shape[0],1)/batch_x.shape[0]
                    reweight_list = reweight_list.to(args.device)
                else:
                    with torch.no_grad():
                        model.eval()
                        reweight_list = adaptive_model((batch_x-X_hat)**2)
                    model.train()
                # print(reweight_list.squeeze(1))
                primal_obj += adaptive_loss(X_hat, batch_x, reweight_list)
            
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj += penalty + l2_reg + l1_reg
            primal_obj.backward()

            return primal_obj
        
        def batch_closure():
            global COUNT
            COUNT += 1
            optimizer.zero_grad()

            primal_obj = torch.tensor(0.).to(args.device)
            loss = torch.tensor(0.).to(args.device)

            for _ , tmp_x in enumerate(train_loader):
                batch_x = tmp_x[0].to(args.device)
                X_hat = model(batch_x)
                primal_obj += squared_loss(X_hat, batch_x)
            
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj += penalty + l2_reg + l1_reg
            primal_obj.backward()
            
            return primal_obj
        
        def w_closure():
            global COUNT
            if COUNT % args.iter_mod == 0 :
                # TODO: add the h_func to control the clip args
                args.clip = torch.sigmoid(model.h_func())
                for i in range (1) :
                    LLL = closure__()
                    space_optimizer.step ()
                    for p in model.fc3.parameters():
                        p.data.clamp_(-args.clip, args.clip)
            COUNT += 1
            optimizer.zero_grad()

            primal_obj = torch.tensor(0.).to(args.device)
            loss_D = torch.tensor(0.).to(args.device)

            for _ , tmp_x in enumerate(train_loader):
                batch_x = tmp_x[0].to(args.device)
                X_hat = model(batch_x)
                x_distribution = model.forward__(batch_x)
                wx_distribution = model.forward__(model(batch_x))
                loss_D += wasserstein_loss(x_distribution, wx_distribution)
                primal_obj += squared_loss(X_hat, batch_x)
                # if COUNT % 10 == 0:
                #     print(loss_D)
            h_val = model.h_func()
            # print(h_val)
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj += penalty + l2_reg + l1_reg + 0.01 * loss_D
            primal_obj.backward()
            
            for p in model.fc3.parameters () :
                p.grad.zero_ ()
            return primal_obj
        
        if run_mode == 1:
            optimizer.step(closure)  # NOTE: updates model in-place
        elif run_mode == 0 :                        # NOTE: the adaptive reweight operation
            optimizer.step(r_closure)
        elif run_mode == 2:
            optimizer.step(batch_closure)
        elif run_mode == 3:
            optimizer.step(w_closure)

        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new


def notears_nonlinear(model: nn.Module,
                      adaptive_model: nn.Module,
                      X: np.ndarray,
                      train_loader: data.DataLoader,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3
                      ):
    rho, alpha, h = 1.0, 0.0, np.inf
    adp_flag = False
    for j in tqdm.tqdm(range(max_iter)):
        rho, alpha, h = dual_ascent_step(model, X, train_loader, lambda1, lambda2,
                                         rho, alpha, h, rho_max, adp_flag, adaptive_model)
        
        if h <= h_tol or rho >= rho_max:
            break
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def hard_mining(data, model, loss_func, ratio = 0.01):
    """
    data: (N_observations, nodes)
    """
    N_sample = data.shape[0]
    model.eval()
    data_hat = model(data)
    loss_col = loss_func(data_hat, data)
    loss_col = torch.sum(loss_col, dim=1)
    loss_col = loss_col.cpu().detach().numpy()
    # 找出最大ratio的loss_col的index
    hard_index_list = np.argsort(loss_col)[::-1][:int(N_sample * ratio)]
    easy_index_list = np.argsort(loss_col)[:int(N_sample * ratio)]
    return hard_index_list, easy_index_list

def main():
    # fangfu

    print('==' * 20)

    import notears.utils as ut
    set_random_seed(args.seed)

    if args.data_type == 'real':
        # X = np.loadtxt('/opt/data2/git_fangfu/JTT_CD/data/sachs.csv', delimiter=',')
        X = np.loadtxt('/opt/data2/git_fangfu/notears/sachs_data/sachs.csv', delimiter=',')
        B_true = np.loadtxt('/opt/data2/git_fangfu/notears/sachs_data/sachs_B_true.csv', delimiter=',')
        model = NotearsMLP(dims=[11, 1], bias=True) # for the real data (sachs)   the nodes of sachs are 11
        adaptive_model = adaptiveMLP(args.batch_size, input_size=X.shape[-1], hidden_size= X.shape[-1] , output_size=1, temperature=args.temperature).to(args.device)

    elif args.data_type == 'synthetic':
        B_true = ut.simulate_dag(args.d, args.s0, args.graph_type)
        X = ut.simulate_nonlinear_sem(B_true, args.n, args.sem_type)
        model = NotearsMLP(dims=[args.d, 10, 1], bias=True) # FIXME: the layer of the Notears MLP
        adaptive_model = adaptiveMLP(args.batch_size, input_size=X.shape[-1], hidden_size= X.shape[-1] , output_size=1, temperature=args.temperature).to(args.device)
    
    elif args.data_type == 'testing':
        B_true = np.loadtxt('testing_B_true.csv', delimiter=',')
        X = np.loadtxt('testing_X.csv', delimiter=',')
        model = NotearsMLP(dims=[args.d ,10, 1], bias=True) # FIXME: the layer of the Notears MLP
        adaptive_model = adaptiveMLP(args.batch_size, input_size=X.shape[-1], hidden_size= X.shape[-1] , output_size=1, temperature=args.temperature).to(args.device)
    
    elif args.data_type == 'sachs_full':
        X = np.loadtxt('/opt/data2/git_fangfu/notears/sachs_data/sachs7466.csv', delimiter=',')
        B_true = np.loadtxt('/opt/data2/git_fangfu/notears/sachs_data/sachs_B_true.csv', delimiter=',')
        model = NotearsMLP(dims=[11, 1], bias=True) # for the real data (sachs)   the nodes of sachs are 11
        adaptive_model = adaptiveMLP(args.batch_size, input_size=X.shape[-1], hidden_size= X.shape[-1] , output_size=1, temperature=args.temperature).to(args.device)
        
    X = torch.from_numpy(X).float().to(args.device)
    model.to(args.device)
    
    # TODO: 将X装入DataLoader
    X_data = data.TensorDataset(X)
    train_loader = data.DataLoader(X_data, batch_size=args.batch_size, shuffle=True)

    W_est = notears_nonlinear(model, adaptive_model, X, train_loader, args.lambda1, args.lambda2)
    assert ut.is_dag(W_est)
    # np.savetxt('W_est.csv', W_est, delimiter=',')
    acc = ut.count_accuracy(B_true, W_est != 0)
    print(acc)
    # 根据args.d和args.s0生成文件夹
    import os
    if not os.path.exists(f'wasserstein/{args.d}_{args.s0}/{args.graph_type}_{args.sem_type}'):
        os.makedirs(f'wasserstein/{args.d}_{args.s0}/{args.graph_type}_{args.sem_type}')
    # 创建该'my_experiment/{args.d}_{args.s0}/{args.graph_type}_{args.sem_type}/{args.seed}.txt'该文件

    if args.data_type == 'synthetic':
        with open(f'wasserstein/{args.d}_{args.s0}/{args.graph_type}_{args.sem_type}/seed_{args.seed}.txt', 'a') as f:
            f.write(f'run_mode: {run_mode}\n')
            f.write(f'observation_num: {args.n}\n')
            f.write(f'without tuning \n')
            if args.run_mode==3:
                f.write(f'batch_size: {args.batch_size}\n')
            f.write(f'dataset_type:{args.data_type}\n')
            f.write(f'acc:{acc}\n')
            f.write('-----------------------------------------------------\n')

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=10)
    main()
