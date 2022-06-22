import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True,help='config file path')
    parser.add_argument('--data_dir', type=str, default='data', help='dataset_path')
    parser.add_argument('--d', type=int, default=5, help='number of nodes')
    parser.add_argument('--s0', type=int, default=9, help='number of edges')
    parser.add_argument('--graph_type', type=str, default='ER', help='graph type')
    parser.add_argument('--sem_type', type=str, default='mlp', help='sem type')
    parser.add_argument('--n', type=int, default=200, help='number of samples')
    parser.add_argument('--lambda1', type=float, default=0.01, help='lambda1')
    parser.add_argument('--lambda2', type=float, default=0.01, help='lambda2')

    return parser