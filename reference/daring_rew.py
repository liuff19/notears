from locally_connected import LocallyConnected
from lbfgsb_scipy import LBFGSBScipy
import torch
import torch.nn as nn
import numpy as np
import argparse
import configargparse
from torch.utils.tensorboard import SummaryWriter
# sys.path.append("/home/heyue/home/Discovery/")
import igraph as ig
import torch.optim as optim
import tqdm as tqdm
import os
def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument("--max_iter", default=5, type=int)
    parser.add_argument("--inner_max_iter", default=1000, type=int)
    parser.add_argument("--h_tol", default=1e-8, type=float)
    parser.add_argument("--rho_max", default=1e16, type=float)
    parser.add_argument("--w_threshold", default=0.3, type=float)
    parser.add_argument("--c_A", default=1, type=float)
    parser.add_argument("--lambda1", default=0.01, type=float)
    parser.add_argument("--lambda2", default=0.01, type=float)
    parser.add_argument("--lambda3", default=0.0, type=float)
    parser.add_argument("--lambda_D", default=0.1, type=float)
    parser.add_argument("--beta", default=1000, type=int)
    parser.add_argument("--group_num", default=5, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--method", default="NOTEARS", type=str)
    parser.add_argument("--n_attributes", default=5, type=int)
    parser.add_argument("--noise_variance_type", default=1, type=int)
    parser.add_argument("--loss_type", default='l2')
    parser.add_argument("--non_linear_type", default=1, type=int)
    parser.add_argument("--batch_size", default=853, type=int)
    parser.add_argument("--sem_type", default="mlp", help = "mlp or mim or gp or gp-add")
    parser.add_argument("--graph_type", default='SF', help="SF/ER/BA")
    parser.add_argument("--s0", default=200, type=int)
    parser.add_argument("--d", default=50, type=int)
    parser.add_argument("--n", default=1200, type=int, help="number of sample observations")
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--scheduler", default=2, type=int)
    parser.add_argument("--data_dir", default='./data/')
    parser.add_argument("--case", default=None, type=str)
    parser.add_argument("--data_type", default='real', type=str,help="synthetic/real")

    # reweight setting for DARING
    parser.add_argument("--reweight", default=True, type=bool, help="whether to reweight the loss")
    parser.add_argument("--reweight_count", default=100, type=int, help='the step to reweight the loss func')
    parser.add_argument("--reweight_gama", default=10, type=int, help='reweight coeeficient')
    parser.add_argument("--reweight_ratio", default=0.15, type=float, help='reweight ratio of the all data')
    return parser


parser = config_parser()
args = parser.parse_args()
# 记录不好的observation 的loss
sp_loss_index = []

# add the tensorboard
prefix =""
if args.reweight:
    prefix = f'{args.method}_ifreweight:{args.reweight}_reweight_cnt:{args.reweight_count}_reweight_gama:{args.reweight_gama}_reweight_ratio:{args.reweight_ratio}_data_type:{args.data_type}_d:{args.d}_n:{args.n}_s0:{args.s0}_graph_type:{args.graph_type}_sem_type:{args.sem_type}_'
else:
    prefix = f'{args.method}_data_type:{args.data_type}_d:{args.d}_n:{args.n}_s0:{args.s0}_graph_type:{args.graph_type}_sem_type:{args.sem_type}_'


writer = SummaryWriter(log_dir = f'/opt/data2/git_fangfu/JTT_CD/logs/d_{args.d}_s_{args.s0}_{args.graph_type}/' + prefix, filename_suffix= prefix)


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
        for l in range(len(dims) - 2):  # extend to MLP
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)
        # fc3: Discriminator #extend to MLP
        layers = []
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
        W = 1 - torch.eye(x.shape[1])
        W = W.reshape(1, W.shape[0], -1).expand(x.shape[0], -1, -1)
        x = x.reshape(x.shape[0], 1, -1).expand(-1, x.shape[1], -1)
        x = x * W
        for _, fc in enumerate(self.fc3):
            if _ != 0:
                x = torch.relu(x)  # [n, d, m1]
            # else :
            # x = torch.nn.functional.dropout (x, 0.5)
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
        M = torch.eye(d) + A / d  # (Yu et al. 2019)
        E = torch.matrix_power(M, d - 1)
        h = torch.trace(E.t() * M).sum() - d
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

class reLoss(nn.Module):
    def __init__(self):
        super(reLoss, self).__init__()

    def forward(self, output, target, reweight_list, gama):
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
        return loss

def squared_loss(output, target, I=0, J=0):
    if I == 0:
        loss = 0.5 * (((output - target) ** 2).mean(0)).sum()
    else:
        N, D = target.shape
        EXY = output.t().mm(target) / N
        EX = output.mean(0).view(-1, 1)
        EY = target.mean(0).view(-1, 1)
        DX = output.std(0).view(-1, 1)
        DY = target.std(0).view(-1, 1)
        L = ((EXY - EX.mm(EY.t())) / (DX.mm(DY.t()))).diagonal().pow(2)
        if J == 0:
            loss = L.sum()
        else:
            loss = (L - L.min().detach()).sum() / D
    return loss

def sample_loss(output, target):
    n = target.shape[0]
    sp_loss = 0.5 / n * ((output - target) ** 2)
    return sp_loss

def kl_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum(torch.log(output / target) + (target / output) - 1)
    return loss

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

def dual_ascent_step(model, X, lambda1, lambda2, lambda3, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    optimizer__ = optim.SGD(model.fc3.parameters(), lr=1e-2, momentum=0.9)
    X_torch = torch.from_numpy(X)
    while rho < rho_max:
        def closure__():
            optimizer__.zero_grad()
            X_hat = model(X_torch)
            xxx = X_torch - X_hat.detach()
            X_R = model.forward__(xxx)
            loss = - squared_loss(X_R, xxx, 1)
            l2_reg = 10 * 0.5 * lambda2 * model.l2_reg__()
            primal_obj = loss + l2_reg
            primal_obj.backward()
            return primal_obj


        def r_closure():
            global COUNT
            global sp_loss_index
            if COUNT % 2 == 0:
                for i in range(1):
                    LLL = closure__()
                    optimizer__.step()
            COUNT += 1

            optimizer.zero_grad()
            X_hat = model(X_torch)
            X_R = model.forward__(X_torch - X_hat)

            if COUNT < args.reweight_count:
                loss = squared_loss(X_hat, X_torch)

            elif COUNT == args.reweight_count:
                loss = squared_loss(X_hat, X_torch)
                sp_loss = sample_loss(X_hat, X_torch)
                # 将sp_loss按照列求平均
                sp_loss = sp_loss.mean(dim=1)

                # 用numpy的方法从sp_loss 中选取loss最大的10%的索引
                sp_loss_index = np.argsort(sp_loss.cpu().detach().numpy())[-int(len(sp_loss) * args.reweight_ratio):]

                # sp_loss_index = np.argsort(sp_loss.cpu().detach().numpy())[:int(len(sp_loss) * args.reweight_ratio)]
                print("ready to reweighting")

            else:
                # loss = reweighted_loss(X_hat, X_torch, sp_loss_index, args.reweight_gama)
                reloss = reLoss()
                loss = reloss(X_hat, X_torch, sp_loss_index, args.reweight_gama)


            lossD = squared_loss(X_R, X_torch - X_hat, 1, 1)

            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg + lambda3 * lossD
            primal_obj.backward()
            for p in model.fc3.parameters():
                p.grad.zero_()
            return primal_obj

        optimizer.step(r_closure)  # NOTE: updates model in-place
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new


def threshold_till_dag(B):
    if ig.Graph.Weighted_Adjacency(B.tolist()).is_dag():
        return B

    nonzero_indices = np.where(B != 0)
    weight_indices_ls = list(zip(B[nonzero_indices], nonzero_indices[0], nonzero_indices[1]))
    sorted_weight_indices_ls = sorted(weight_indices_ls, key=lambda tup: abs(tup[0]))

    for weight, i, j in sorted_weight_indices_ls:
        B[i, j] = 0
        if ig.Graph.Weighted_Adjacency(B.tolist()).is_dag():
            break

    return B


def notears_nonlinear(model: nn.Module,
                      X: np.ndarray,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      lambda3: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in tqdm.tqdm(range(max_iter)):
        rho, alpha, h = dual_ascent_step(model, X, lambda1, lambda2, lambda3,
                                         rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break

    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    W_est = threshold_till_dag(W_est)
    return W_est


def get_result(X, B_true):
    torch.set_default_dtype(torch.float32)
    np.set_printoptions(precision=3)

    global COUNT
    COUNT = 0

    model = NotearsMLP(dims=[X.shape[1], 1], bias=True)
    W_est = notears_nonlinear(model, X, lambda1=args.lambda1, lambda2 = args.lambda2, lambda3=args.lambda3)  # if lambda3==0, it equals to NOTEARS

    return W_est

def main():
    parser = config_parser()
    args = parser.parse_args()
    
    # print args onebyone
    for argname, argval in vars(args).items():
        print(f'{argname.replace("_", " ").capitalize()}: {argval}')
    

    import daring_utils as ut
    # seed = random.randint(1, 10000)
    # print(seed)
    ut.set_random_seed(1000)

    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)
    # 随机种子固定
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    global COUNT
    COUNT = 0

    # real data or synthetic data
    if args.data_type == 'real':
        X = np.loadtxt('/opt/data2/git_fangfu/JTT_CD/data/sachs.csv', delimiter=',')
        B_true = np.loadtxt('/opt/data2/git_fangfu/JTT_CD/data/sachs_B_true.csv', delimiter=',')
    
    if args.data_type == 'synthetic':
        B_true = ut.simulate_dag(args.d, args.s0, args.graph_type)
        # np.savetxt('W_true.csv', B_true, delimiter=',')
        X = ut.simulate_nonlinear_sem(B_true, args.n, args.sem_type)
        # 
    
    
    ##==================================================================================================================

    # synthetic data from the yuwang data
    # W_true = ut.load_nonlinear_graph(args)

    # B_true = (W_true != 0).astype(int)

    # X = ut.load_nonlinear_data(args, **kwargs)

    X = torch.from_numpy(X).float().to('cpu')
    X = X.unsqueeze(0)
    X = [torch.tensor(x, dtype=torch.float32) for x in X]
    X = torch.cat(X)
    # 将X保存为X.csv
    np.savetxt('X.csv', X.cpu().numpy(), delimiter=',')
    # 将B_true保存为B_true.csv
    np.savetxt('B_true.csv', B_true, delimiter=',')
    W_est = get_result(X.numpy(), B_true)


    print("X.Shape: ", X.shape)

    acc = ut.count_accuracy(B_true, W_est != 0)
    print(acc)
    print(f"total_count:{COUNT}")

    # save the result
    method = 'NOTEARS'
    if args.lambda3 == 0:
        method = 'NOTEARS'
    else:
        method = 'Daring'

    record_path = f'/opt/data2/git_fangfu/JTT_CD/logs/d{args.d}_s{args.s0}_{args.graph_type}.txt'
    # 如果没有就创建
    if not os.path.exists(record_path):
        os.mknod(record_path)

    with open(record_path, 'a') as f:

        f.write(f'method:{method}\n')

        if args.data_type == 'real':
            f.write(f'data_type:{args.data_type}\n')
            f.write(f'd(nodes):11\n')
            f.write(f's0(edges):17\n')

        if args.data_type == 'synthetic':
            f.write(f'data_type:{args.data_type}\n')
            f.write(f'd(nodes):{args.d}\n')
            f.write(f's0(edges):{args.s0}\n')
            f.write(f'graph_type:{args.graph_type}\n')
            f.write(f'n(samples):{args.n}\n')
            f.write(f'sem_type:{args.sem_type}\n')

        f.write(f'lambda1:{args.lambda1}\n')
        f.write(f'lambda2:{args.lambda2}\n')
        f.write(f'reweight:{args.reweight}\n')

        if args.reweight:
            f.write(f'reweight_cnt:{args.reweight_count}\n')
            f.write(f'reweight_ratio:{args.reweight_ratio}\n')
            f.write(f'total_step:{COUNT}\n')
            f.write(f'reweight_gama:{args.reweight_gama}\n')

        f.write(f'acc:{acc}\n')
        f.write('-----------------------------------------------------\n')
        f.write('-----------------------------------------------------\n')
    
        # 用tensorboard记录这些hyperparameters
    writer.add_hparams({'method': method, 'data_type':args.data_type, 'lambda1': args.lambda1, 'lambda2':args.lambda2, 'reweight': args.reweight, 'reweight_cnt': args.reweight_count,'reweight_ratio':args.reweight_ratio,'reweight_gama':args.reweight_gama, 'd':args.d, 's0':args.s0,'n':args.n,'sem_type':args.sem_type,'graph_type':args.graph_type}, {'hparam/total_iter_cnt':COUNT,'hparam/fdr': acc['fdr'], 'hparam/tpr': acc['tpr'], 'hparam/fpr': acc['fpr'], 'hparam/shd': acc['shd'],'hparam/nnz': acc['nnz']})

if __name__ == '__main__':
    main()