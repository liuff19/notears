import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True,help='config file path')
    parser.add_argument('--device', default='cpu', help='cuda or cpu')

    parser.add_argument("--s0", default=40, type=int)
    parser.add_argument("--d", default=10, type=int) # 如果是10 下面的n就是1000, 如果是20就是2000, 如果d>50, n最好加大
    parser.add_argument("--n", default=1000, type=int)
    parser.add_argument("--sem_type", default="mlp", choices=["gp-add","mlp", "gp", "mim"])
    parser.add_argument("--graph_type", default='ER', choices=['SF', 'ER', 'BA'])


    parser.add_argument('--data_dir', type=str, default='data', help='dataset_path')
    parser.add_argument('--seed', type=int, default=7, help='random seed')

    parser.add_argument('--lambda1', type=float, default=0.01, help='lambda1')
    parser.add_argument('--lambda2', type=float, default=0.01, help='lambda2')
    parser.add_argument('--reweight', action='store_true', help='if reweight')
    parser.add_argument('--beta', type=float, default=0.9, help='beta')
    
    parser.add_argument("--w_threshold", default=0.3, type=float)
    parser.add_argument("--data_type", default='synthetic', type=str, help = 'real or synthetic', choices=['real', 'synthetic','testing', 'sachs_full'])


    # TODO: add the arguments for adapitve reweight ， 默认fit好的参数，t = 20, 10, batch=200, adaptive_epoch=10
    # add the batch_size
    parser.add_argument('--batch_size', type=int, default=200, help='batch_size')
    parser.add_argument('--reweight_epoch', type=int, default=0, help='the epoch begin to reweight')
    
    parser.add_argument('--temperature', type=int, default=20, help='softmax_tmperature')
    parser.add_argument("--adaptive_epoch", default=10, type=int, help="number of iterations for adaptive reweight")
    parser.add_argument("--adaptive_lr", default=0.0001, type=float, help="learning rate for adaptive reweight")
    parser.add_argument("--adaptive_lambda", default=0.001, type=float, help="adaptive lambda for l1 regularization")    
    
    # TODO: add arguments for wasserstein loss
    parser.add_argument('--run_mode', type = int, default=3, help=' 0: reweight + batch ; 1: baseline; 2: batch + baseline; 3: wasserstein')
    parser.add_argument('--iter_mod', type= int, default = 2, help='the iteration mod for wassertein')
    parser.add_argument('--clip', type= float, default =0.01, help='the clip for Lipshitz')
    parser.add_argument('--space_lr', type=int, default =0.001, help='the learning rate of space optimizer')
    
    return parser

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    print(args)
