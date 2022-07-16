# from notears.locally_connected import LocallyConnected
# from notears.lbfgsb_scipy import LBFGSBScipy
# from notears.trace_expm import trace_expm
from locally_connected import LocallyConnected
from trace_expm import trace_expm
import torch
import torch.nn as nn
from lbfgsb_scipy import LBFGSBScipy
import numpy as np
import tqdm as tqdm
from runhelper import *
from loss_func import *
import random
import time
import igraph as ig

import utils as ut

COUNT = 0
reweight_cnt = 800
reweight_ratio = 0.2
reweight_gama = 1000
ifreweight = 1
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

def reweighted_loss(output, target, reweight_list, gama):
    n = target.shape[0]

    # 计算方式0
    re_output = output[reweight_list]
    re_target = target[reweight_list]
    re_R = re_output - re_target

    keep_list = [i for i in range(output.shape[0]) if i not in reweight_list]
 
    keep_output = output[keep_list]
    keep_target = target[keep_list]
    R = keep_output - keep_target

    reweight_len = len(reweight_list)
    assert reweight_len == re_output.shape[0]
    loss = 0.5 * gama*( re_R** 2).sum()/(re_R.shape[0]) + 0.5 * (R ** 2).sum()/(R.shape[0])

    # re_matrix = torch.eye(n,n).to(target.device)
    # for idx in reweight_list:
    #     re_matrix[idx,idx] = gama
    # R = output-target
    # # 计算方式1
    # loss = 0.5 / n * torch.sum(torch.matmul(re_matrix, R) ** 2)
    # 计算方式2
    # loss = torch.sum(torch.matmul(re_matrix, R) ** 2)
    # 计算方式3
    # loss = 0.5 / (n+(gama-1)*reweight_len)* torch.sum(torch.matmul(re_matrix, R) ** 2)
    return loss

def sample_loss(output, target):
    n = target.shape[0]
    sp_loss = 0.5 / n * ((output - target) ** 2)
    return sp_loss


def dual_ascent_step(model, X, lambda1, lambda2, rho, alpha, h, rho_max):
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
            return primal_obj

        def r_closure():
            global COUNT
            COUNT += 1
            optimizer.zero_grad()
            X_hat = model(X)
            sp_idx = []
            if COUNT < reweight_cnt:
                loss = squared_loss(X_hat, X)
            elif COUNT == reweight_cnt:
                loss = squared_loss(X_hat, X)
                sp_loss = sample_loss(X_hat, X)
                sp_loss = sp_loss.mean(dim=1)
                _, sp_idx = torch.topk(sp_loss, int(sp_loss.shape[0] * reweight_ratio))
                print("ready to reweighting")
            else:
                loss = reweighted_loss(X_hat, X, sp_idx, reweight_gama)

            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            primal_obj.backward()
            return primal_obj

        if ifreweight:
            optimizer.step(r_closure)  # NOTE: updates model in-place
        else:
            optimizer.step(closure) 

        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new


def notears_nonlinear(model: nn.Module,
                      X: np.ndarray,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    start = time.time()
    tloop = tqdm.tqdm(range(max_iter))
    for _ in tloop:
        rho, alpha, h = dual_ascent_step(model, X, lambda1, lambda2,
                                         rho, alpha, h, rho_max)
        with torch.no_grad():
            loss = squared_loss(model(X), X)

        tloop.set_postfix(loss = loss.item())
        if h <= h_tol or rho >= rho_max:
            break
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    end = time.time()
    print("time:", end - start)
    return W_est

def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def compute_acc(model):
    parser = config_parser()
    args = parser.parse_args()
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < args.w_threshold] = 0
    _, W_true = getdata(args)

    if not is_dag(W_est):
        # print("!!! Result is not DAG, Now cut the edge until it's DAG")
        # Find the smallest threshold that removes all cycle-inducing edges
        thresholds = np.unique(W_est)
        for step, t in enumerate(thresholds):
            # print("Edges/thresh", W_est.sum(), t)
            to_keep = torch.Tensor(W_est > t + 1e-8).numpy()
            W_est = W_est * to_keep

            if is_dag(W_est):
                break
    acc = ut.count_accuracy(W_true, W_est != 0)
    return acc

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def getdata(args):
    if args.data_type == 'real':
        X = np.loadtxt('/opt/data2/git_fangfu/JTT_CD/data/sachs.csv', delimiter=',')
        B_true = np.loadtxt('/opt/data2/git_fangfu/JTT_CD/data/sachs_B_true.csv', delimiter=',')
    elif args.data_type == 'synthetic':
        B_true = ut.simulate_dag(args.d, args.s0, args.graph_type)
        X = ut.simulate_nonlinear_sem(B_true, args.n, args.sem_type)

    return X, B_true

def main():
    # fangfu
    parser = config_parser()
    args = parser.parse_args()
    print(args)
    print('==' * 20)

    set_random_seed(args.seed)


    if args.data_type == 'real':
        X = np.loadtxt('/opt/data2/git_fangfu/JTT_CD/data/sachs.csv', delimiter=',')
        B_true = np.loadtxt('/opt/data2/git_fangfu/JTT_CD/data/sachs_B_true.csv', delimiter=',')
        model = NotearsMLP(dims=[11,10, 1], bias=True) # for the real data (sachs)   the nodes of sachs are 11

    elif args.data_type == 'synthetic':
        B_true = ut.simulate_dag(args.d, args.s0, args.graph_type)
        X = ut.simulate_nonlinear_sem(B_true, args.n, args.sem_type)
        model = NotearsMLP(dims=[args.d,10, 1], bias=True) # for the synthetic data
    
    X = torch.from_numpy(X).float().to(args.device)
    model.to(args.device)

    W_est = notears_nonlinear(model, X, args.lambda1, args.lambda2)


    if not is_dag(W_est):
        print("!!! Result is not DAG, Now cut the edge until it's DAG")
        thresholds = np.unique(W_est)
        for step, t in enumerate(thresholds):
            to_keep = torch.Tensor(W_est > t + 1e-8).numpy()
            W_est = W_est * to_keep

            if is_dag(W_est):
                break


    acc = ut.count_accuracy(B_true, W_est != 0)
    print(acc)
    print(f"total_count:{COUNT}")

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=10)
    main()
