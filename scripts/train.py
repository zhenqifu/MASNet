import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch
import torch.optim as optim
import glob
import random
import numpy as np
import torch.optim.lr_scheduler as lrs
from configs.config import get_gonfig
from data.data_loader import train_loader
from data.FDA import FDA_source_to_target
from utils.loss import loss_task
from models.masnet import Net
from utils.log import get_logger, save_ckpt
from tqdm import tqdm


def seed_torch(seed=20222022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_torch()

def main():
    # get configs
    args = get_gonfig()
    train(args)


def train(args):
    exp_name = args['exp_name']
    # --------------------------------------loading data------------------------------------------
    img_name_list = glob.glob(args['Train']['tra_image_dir'] + '*' + '.jpg')
    lab_name_list = []
    for img_path in img_name_list:
        img_name = img_path.split("/")[-1]
        imidx = img_name.split(".")[0]
        lab_name_list.append(args['Train']['tra_lbl_dir'] + imidx + '.jpg')  # MAS3K:png  RMAS:jpg 

    logger = get_logger(exp_name)
    logger.info("The code of this experiment has been saved to {}".format(os.path.abspath('..')))
    logger.info("---")
    logger.info(exp_name)
    logger.info("train images: {}".format(len(img_name_list)))
    logger.info("train labels: {}".format(len(lab_name_list)))
    logger.info("---")

    print("-------------------")
    print(exp_name)
    print("train images: {}".format(len(img_name_list)))
    print("train labels: {}".format(len(lab_name_list)))
    print("-------------------")

    train_dataloader = train_loader(img_name_list, lab_name_list, args['Train']['batch_size'], args['Train']['train_size'])

    # ----------------------------------define model --------------------------------------------
    # define the net
    net_name = args['Train']['net']
    net = Net()
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    loss_mse = torch.nn.MSELoss()

    logger.info("Total number of net paramerters {}".format(sum(x.numel() for x in net.parameters())))

    # ------------------------------ define optimizer ------------------------------------------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=args['Train']['optimizer_lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    milestones = []
    for i in range(1, args['Train']['epoch_num']):
        if i % 100 == 0:
            milestones.append(i)
    scheduler = lrs.MultiStepLR(optimizer, milestones, 0.5)


    # ======================================= training process ===================================================

    logger.info("---start training...")
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    ite_num4val = 0

    # ----------------------------------------training stage----------------------------------------------------------
    for epoch in range(0, args['Train']['epoch_num']):
        net.train()

        bar_format = '{desc}{percentage:3.0f}%|{bar}|{n_fmt:.5s}/{total_fmt}[{elapsed}<{remaining}{postfix}]'
        loop = tqdm(enumerate(train_dataloader, 1), total=len(train_dataloader), ncols=100, bar_format=bar_format)

        for i, data in loop:
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1
            inputs, labels, inputs_sod = data[0], data[1], data[2]

            inputs = inputs.cuda()
            labels = labels.cuda()
            inputs2 = inputs_sod.cuda()

            B = inputs.size(0)
            perm = torch.randperm(B)
            inputs_shuffling = inputs[perm]
            inputs2 = FDA_source_to_target(inputs2, inputs_shuffling)
            inputs2 = inputs2.cuda()

            output = net(inputs)
            output2 = net(inputs2)
            loss1 = loss_task(output, labels)
            loss2 = loss_mse(output2, output)
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()           
            
 
    # ---------------------------------------logging info-----------------------------------------------------
            logger.info("%s  :[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, lr: %6f" % (net_name,
            epoch + 1, args['Train']['epoch_num'], (i + 1) * args['Train']['batch_size'], 
            len(img_name_list), ite_num, running_loss / ite_num4val, optimizer.param_groups[0]['lr']))  # writing to log file

            loop.set_description(f"{ net_name } : Epoch [{ epoch+1 }/{ args['Train']['epoch_num'] }]")  # Set the description of tqdm bar
            loop.set_postfix(train_loss=f'{running_loss / ite_num4val:.3f}')   # Set the postfix of tqdm bar, edit as needed
            
            del loss

        scheduler.step()
    # ---------------------------------------save checkpoint--------------------------------------------------
            
        if (epoch+1) % 2 == 0:  
            ckpt = {
                    "net": net.state_dict(),
                    "current_epoch": epoch,
                    "ite_num": ite_num
                    }
            save_ckpt(ckpt, name=net_name, epoch=epoch)

            running_loss = 0.0
            ite_num4val = 0

        loop.close()

    print('\033[1;33m-------------Congratulations! Training Done!!!-------------\033[0m')
    logger.info('-------------Congratulations! Training Done!!!-------------')


if __name__ == '__main__':
    main()
