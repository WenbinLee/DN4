import torch
import torch.nn as nn
from torch.nn import init
import functools
import pdb
import math

''' 
	
	This Network is designed for Few-Shot Learning Problem. 

'''


###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('Linear') != -1:
		init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.xavier_normal_(m.weight.data, gain=0.02)
	elif classname.find('Linear') != -1:
		init.xavier_normal_(m.weight.data, gain=0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
	classname = m.__class__.__name__
	print(classname)
	if classname.find('Conv') != -1:
		init.orthogonal_(m.weight.data, gain=1)
	elif classname.find('Linear') != -1:
		init.orthogonal_(m.weight.data, gain=1)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
	print('initialization method [%s]' % init_type)
	if init_type == 'normal':
		net.apply(weights_init_normal)
	elif init_type == 'xavier':
		net.apply(weights_init_xavier)
	elif init_type == 'kaiming':
		net.apply(weights_init_kaiming)
	elif init_type == 'orthogonal':
		net.apply(weights_init_orthogonal)
	else:
		raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
	elif norm_type == 'none':
		norm_layer = None
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer



def define_DN4Net(pretrained=False, model_root=None, which_model='Conv64', norm='batch', init_type='normal', use_gpu=True, **kwargs):
	DN4Net = None
	norm_layer = get_norm_layer(norm_type=norm)

	if use_gpu:
		assert(torch.cuda.is_available())

	if which_model == 'Conv64F':
		DN4Net = FourLayer_64F(norm_layer=norm_layer, **kwargs)
	elif which_model == 'ResNet256F':
		net_opt = {'userelu': False, 'in_planes':3, 'dropout':0.5, 'norm_layer': norm_layer} 
		DN4Net = ResNetLike(net_opt)
	else:
		raise NotImplementedError('Model name [%s] is not recognized' % which_model)
	init_weights(DN4Net, init_type=init_type)

	if use_gpu:
		DN4Net.cuda()

	if pretrained:
		DN4Net.load_state_dict(model_root)

	return DN4Net


def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)



##############################################################################
# Classes: FourLayer_64F
##############################################################################

# Model: FourLayer_64F 
# Input: One query image and a support set
# Base_model: 4 Convolutional layers --> Image-to-Class layer  
# Dataset: 3 x 84 x 84, for miniImageNet
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21


class FourLayer_64F(nn.Module):
	def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=5, neighbor_k=3):
		super(FourLayer_64F, self).__init__()

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		self.features = nn.Sequential(                              # 3*84*84
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),
			nn.MaxPool2d(kernel_size=2, stride=2),                  # 64*42*42

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),
			nn.MaxPool2d(kernel_size=2, stride=2),                  # 64*21*21

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),                                # 64*21*21

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),                                # 64*21*21
		)
		
		self.imgtoclass = ImgtoClass_Metric(neighbor_k=neighbor_k)  # 1*num_classes



	def forward(self, input1, input2):

		# extract features of input1--query image
		q = self.features(input1)

		# extract features of input2--support set
		S = []
		for i in range(len(input2)):
			support_set_sam = self.features(input2[i])
			B, C, h, w = support_set_sam.size()
			support_set_sam = support_set_sam.permute(1, 0, 2, 3)
			support_set_sam = support_set_sam.contiguous().view(C, -1)
			S.append(support_set_sam)

		x = self.imgtoclass(q, S) # get Batch*num_classes

		return x



#========================== Define an image-to-class layer ==========================#


class ImgtoClass_Metric(nn.Module):
	def __init__(self, neighbor_k=3):
		super(ImgtoClass_Metric, self).__init__()
		self.neighbor_k = neighbor_k


	# Calculate the k-Nearest Neighbor of each local descriptor 
	def cal_cosinesimilarity(self, input1, input2):
		B, C, h, w = input1.size()
		Similarity_list = []

		for i in range(B):
			query_sam = input1[i]
			query_sam = query_sam.view(C, -1)
			query_sam = torch.transpose(query_sam, 0, 1)
			query_sam_norm = torch.norm(query_sam, 2, 1, True)    

			if torch.cuda.is_available():
				inner_sim = torch.zeros(1, len(input2)).cuda()

			for j in range(len(input2)):
				support_set_sam = input2[j]
				support_set_sam_norm = torch.norm(support_set_sam, 2, 0, True)

				# cosine similarity between a query sample and a support category
				innerproduct_matrix = query_sam@support_set_sam/(query_sam_norm@support_set_sam_norm)

				# choose the top-k nearest neighbors
				topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 1)
				inner_sim[0, j] = torch.sum(topk_value)

			Similarity_list.append(inner_sim)

		Similarity_list = torch.cat(Similarity_list, 0)    

		return Similarity_list 


	def forward(self, x1, x2):

		Similarity_list = self.cal_cosinesimilarity(x1, x2)

		return Similarity_list




##############################################################################
# Classes: ResNetLike
##############################################################################

# Model: ResNetLike 
# Refer to: https://github.com/gidariss/FewShotWithoutForgetting
# Input: One query image and a support set
# Base_model: 4 ResBlock layers --> Image-to-Class layer  
# Dataset: 3 x 84 x 84, for miniImageNet
# Filters: 64->96->128->256



class ResBlock(nn.Module):
	def __init__(self, nFin, nFout):
		super(ResBlock, self).__init__()

		self.conv_block = nn.Sequential()
		self.conv_block.add_module('BNorm1', nn.BatchNorm2d(nFin))
		self.conv_block.add_module('LRelu1', nn.LeakyReLU(0.2))
		self.conv_block.add_module('ConvL1', nn.Conv2d(nFin,  nFout, kernel_size=3, padding=1, bias=False))
		self.conv_block.add_module('BNorm2', nn.BatchNorm2d(nFout))
		self.conv_block.add_module('LRelu2', nn.LeakyReLU(0.2))
		self.conv_block.add_module('ConvL2', nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))
		self.conv_block.add_module('BNorm3', nn.BatchNorm2d(nFout))
		self.conv_block.add_module('LRelu3', nn.LeakyReLU(0.2))
		self.conv_block.add_module('ConvL3', nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))

		self.skip_layer = nn.Conv2d(nFin, nFout, kernel_size=1, stride=1)


	def forward(self, x):
		return self.skip_layer(x) + self.conv_block(x)



class ResNetLike(nn.Module):
	def __init__(self, opt, neighbor_k=3):
		super(ResNetLike, self).__init__()

		self.in_planes = opt['in_planes']
		self.out_planes = [64, 96, 128, 256]
		self.num_stages = 4

		if type(opt['norm_layer']) == functools.partial:
			use_bias = opt['norm_layer'].func == nn.InstanceNorm2d
		else:
			use_bias = opt['norm_layer'] == nn.InstanceNorm2d


		if type(self.out_planes) == int:
			self.out_planes = [self.out_planes for i in range(self.num_stages)]

		assert(type(self.out_planes)==list)
		assert(len(self.out_planes)==self.num_stages)
		num_planes = [self.out_planes[0],] + self.out_planes
		userelu = opt['userelu'] if ('userelu' in opt) else False
		dropout = opt['dropout'] if ('dropout' in opt) else 0

		self.feat_extractor = nn.Sequential()
		self.feat_extractor.add_module('ConvL0', nn.Conv2d(self.in_planes, num_planes[0], kernel_size=3, padding=1))
		
		for i in range(self.num_stages):
			self.feat_extractor.add_module('ResBlock'+str(i), ResBlock(num_planes[i], num_planes[i+1]))
			if i<self.num_stages-2:
				self.feat_extractor.add_module('MaxPool'+str(i), nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

		self.feat_extractor.add_module('ReluF1', nn.LeakyReLU(0.2, True))                # get Batch*256*21*21
	

		self.imgtoclass = ImgtoClass_Metric(neighbor_k=neighbor_k)    # Batch*num_classes


		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()


	def forward(self, input1, input2):
	
		# extract features of input1--query image
		q = self.feat_extractor(input1)

		# extract features of input2--support set
		S = []
		for i in range(len(input2)):
			support_set_sam = self.feat_extractor(input2[i])
			B, C, h, w = support_set_sam.size()
			support_set_sam = support_set_sam.permute(1, 0, 2, 3)
			support_set_sam = support_set_sam.contiguous().view(C, -1)
			S.append(support_set_sam)


		x = self.imgtoclass(q, S)   # get Batch*num_classes

		return x
