import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import glob
import numpy as np
from PIL import Image
from metrics import *
from models.masnet import Net
from collections import OrderedDict
from configs.config import get_gonfig
from data.data_loader import test_loader
from thop import profile

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)
	dn = (d-mi)/(ma-mi)
	return dn


if __name__ == '__main__':
	# --------- 1. get image path and name ---------
	args = get_gonfig('../config.yaml')

	test_img_name_list = glob.glob(args['Test']['test_image_dir'] + '*' + '.jpg')
	testset_name = args['Test']['testset_name']
	prediction_dir = '../eval_results/' + testset_name + '/'
	if not os.path.exists(prediction_dir):
		os.makedirs(prediction_dir)

	net_dir = args['Test']['ckpt']
	ckpt = torch.load(net_dir)

	# --------- 2. dataloader ---------
	test_loader = test_loader(test_img_name_list, args['Test']['batch_size'], args['Test']['test_size'])
	# --------- 3. model define ---------

	print("...load Net...")
	net = Net()

	state = ckpt['net']
	new_state = OrderedDict()
	for k, v in state.items():
		name = k[7:]
		new_state[name] = v

	net.load_state_dict(new_state)
	net.cuda()
	net.eval()

	# --------- 4. inference for each image ---------
	for i_test, data_test in enumerate(test_loader):
	
		print("inferencing:", test_img_name_list[i_test].split("/")[-1])
	
		inputs_test = data_test[0]
		inputs_test = inputs_test
	
		if torch.cuda.is_available():
			inputs_test = inputs_test.cuda()
		else:
			inputs_test = inputs_test
	
		d0 = net(inputs_test)

		# flops, params = profile(net, (inputs_test,))
		# print('flops: ', flops, 'params: ', params)

		pred = d0[:, 0, :, :]
		pred = normPRED(pred)

		pred = pred.squeeze().cpu().detach().numpy()
		mask = Image.fromarray((pred * 255).astype(np.uint8))
		name = test_img_name_list[i_test].split("/")[-1]
		name = name.split(".")[0]
		mask.save(prediction_dir+'{}'.format(name+'.png')) 
	
	metrics(args['Test']['test_lbl_dir'], prediction_dir)




