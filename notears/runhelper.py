import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True,help='config file path')
    parser.add_argument('--device', default='cpu', help='cuda or cpu')

    parser.add_argument("--s0", default=100, type=int)
    parser.add_argument("--d", default=25, type=int)
    parser.add_argument("--n", default=1000, type=int)
    parser.add_argument("--sem_type", default="mlp", choices=["gp-add","mlp", "gp", "mim"])
    parser.add_argument("--graph_type", default='SF')


    parser.add_argument('--data_dir', type=str, default='data', help='dataset_path')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    parser.add_argument('--lambda1', type=float, default=0.01, help='lambda1')
    parser.add_argument('--lambda2', type=float, default=0.01, help='lambda2')
    parser.add_argument('--reweight', action='store_true', help='if reweight')
    parser.add_argument('--beta', type=float, default=0.9, help='beta')
    parser.add_argument('--reweight_epoch', type=int, default=3, help='the epoch begin to reweight')

    parser.add_argument("--data_type", default='synthetic', type=str, help = 'real or synthetic')
    return parser