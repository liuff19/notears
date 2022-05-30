import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True,help='config file path')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory')