from notears.locally_connected import LocallyConnected
from notears.trace_expm import trace_expm
import torch
import torch.nn as nn
from notears.lbfgsb_scipy import LBFGSBScipy
import numpy as np
import tqdm as tqdm
from notears.runhelper import config_parser
from notears.loss_func import *
import random
import time
import igraph as ig
import notears.utils as ut

import torch.utils.data as data
from adaptive_model.adapModel import adaptiveMLP
from adaptive_model.adapModel import adap_reweight_step



COUNT = 0
IF_baseline = 0

parser = config_parser()
args = parser.parse_args()
print(args)
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

# TODO: create the adaptive_loss function
def adaptive_loss(output, target, reweight_idx):
    R = output-target
    reweight_matrix = torch.diag(reweight_idx).to(args.device)
    # loss = 0.5 * torch.sum(torch.matmul(reweight_matrix, R))
    loss = 0.5 * torch.sum(torch.matmul(reweight_matrix, R**2))
    return loss

def dual_ascent_step(model, X, train_loader, lambda1, lambda2, rho, alpha, h, rho_max, adp_flag, adaptive_model):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    # X_torch = torch.from_numpy(X)
    while rho < rho_max:
        
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
            # if COUNT % 100 == 0:
            #     print(f"{primal_obj}: {primal_obj.item():.4f}; count: {COUNT}")
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
                # TODO: the adaptive loss should add here
                if adp_flag == False:
                    reweight_list = torch.ones(batch_x.shape[0],1)/batch_x.shape[0]
                    reweight_list = reweight_list.to(args.device)
                # else:
                #     with torch.no_grad():
                #         reweight_list = adaptive_model(batch_x)
                loss += squared_loss(X_hat, batch_x)
                # primal_obj += squared_loss(X_hat, batch_x)
            
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj += loss + penalty + l2_reg + l1_reg
            primal_obj.backward()
            # if COUNT % 100 == 0:
            #     print(f"{primal_obj}: {primal_obj.item():.4f}; count: {COUNT}")
            return primal_obj

        if IF_baseline:
            optimizer.step(closure)  # NOTE: updates model in-place
        else:                        # NOTE: the adaptive reweight operation
            optimizer.step(r_closure)

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
        if j > args.reweight_epoch:
            print("Re-weighting")
            # TODO: reweight operation here
            rho, alpha, h = dual_ascent_step(model, X, train_loader, lambda1, lambda2,
                                         rho, alpha, h, rho_max, adp_flag, adaptive_model)
        else:
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

def main():
    # fangfu

    print('==' * 20)

    import notears.utils as ut
    set_random_seed(args.seed)

    if args.data_type == 'real':
        X = np.loadtxt('/opt/data2/git_fangfu/JTT_CD/data/sachs.csv', delimiter=',')
        B_true = np.loadtxt('/opt/data2/git_fangfu/JTT_CD/data/sachs_B_true.csv', delimiter=',')
        model = NotearsMLP(dims=[11, 1], bias=True) # for the real data (sachs)   the nodes of sachs are 11
        adaptive_model = adaptiveMLP(input_size=X.shape[-1], hidden_size= X.shape[-1] , output_size=1).to(args.device)

    elif args.data_type == 'synthetic':
        B_true = ut.simulate_dag(args.d, args.s0, args.graph_type)
        X = ut.simulate_nonlinear_sem(B_true, args.n, args.sem_type)
        model = NotearsMLP(dims=[args.d ,10, 1], bias=True) # FIXME: the layer of the Notears MLP
        adaptive_model = adaptiveMLP(input_size=X.shape[-1], hidden_size= X.shape[-1] , output_size=1).to(args.device)

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
    if args.reweight:
        print('reweighting')

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=10)
    main()
