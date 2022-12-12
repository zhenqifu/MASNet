import logging
import os
import datetime
import yaml
import glob
import shutil
import torch
from tensorboardX import SummaryWriter

def get_exp_name():
    exp_name="exp_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(os.path.join("./runs", exp_name)):
        os.makedirs(os.path.join("./runs", exp_name))
        os.makedirs(os.path.join("./runs", exp_name,"checkpoints"))
        os.makedirs(os.path.join("./runs", exp_name,"eval_results"))
        os.makedirs(os.path.join("./runs", exp_name,"logs"))
        os.makedirs(os.path.join("./runs", exp_name,"runs"))
        os.makedirs(os.path.join("./runs", exp_name,"scripts"))
    return exp_name

# def summary_model(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
#     result, params_info = summary_string(model, input_size, batch_size, device, dtypes)
#     return result
def summary_writer(exp_name):
    # writer=SummaryWriter(os.path.join("./runs", exp_name,"runs"))
    writer=SummaryWriter(os.path.join("../runs"))
    return writer

def get_logger(exp_name, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # fh = logging.FileHandler("./runs/"+exp_name+"/logs/log", "w") #输出到日志文件
    fh = logging.FileHandler("../logs/log", "w") #输出到日志文件
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # sh = logging.StreamHandler() #输出到控制台
    # sh.setFormatter(formatter)
    # logger.addHandler(sh)

    return logger


def save_config(exp_name,args):
    args['exp_name']=exp_name
    with open(os.path.join("./runs", exp_name, "config.yaml"), "w") as cfg:
        yaml.dump(args, cfg)
    
    scripts_to_save=glob.glob('*.py')
    for script in scripts_to_save:
        dst_file = os.path.join("./runs",exp_name, "scripts", os.path.basename(script))
        shutil.copyfile(script, dst_file)
    despath=os.path.join("./runs",exp_name, "scripts")
    oripath_1="./configs/"
    oripath_2="./data/"
    oripath_3="./models/"
    oripath_4="./utils/"
    oripath_5="./watchmen/"
    shutil.copytree(oripath_1, despath+"/configs")
    shutil.copytree(oripath_2, despath+"/data")
    shutil.copytree(oripath_3, despath+"/models")
    shutil.copytree(oripath_4, despath+"/utils")
    shutil.copytree(oripath_5, despath+"/watchmen")

def save_ckpt(ckpt,**kwargs):
    torch.save(ckpt,os.path.join("../checkpoints/", f"{kwargs['name']}", f"epoch_{kwargs['epoch']}.pth"))
