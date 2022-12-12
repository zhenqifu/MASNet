import yaml

config_file='../config.yaml'
def get_gonfig(config_file=config_file):
    with open(config_file,'r') as cfg:
        args = yaml.load(cfg, Loader=yaml.FullLoader)
    return args
    



