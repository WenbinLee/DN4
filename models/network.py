import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import functools
import random
import pdb
import copy
import sys
sys.dont_write_bytecode = True



# ============================ Backbone & Classifier ===============================
import models.backbone as backbone
import models.classifier as classifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
# ==================================================================================
''' 
	All models consist of two parts: backbone module and classifier module.
'''


###############################################################################
# Functions
###############################################################################

encoder_dict = dict(
			Conv64F       = backbone.Conv64F,
			Conv64F_Local = backbone.Conv64F_Local,
			ResNet10      = backbone.ResNet10,
			ResNet12      = backbone.ResNet12,
			SeResNet12    = backbone.SeResNet12,
			ResNet18      = backbone.ResNet18,
			ResNet34      = backbone.ResNet34,
			ResNet50      = backbone.ResNet50,
			ResNet101     = backbone.ResNet101) 


classifier_dict = dict(
			ProtoNet      = classifier.Prototype_Metric,
			DN4           = classifier.ImgtoClass_Metric) 



def weights_init_normal(m):
	classname = m.__class__.__name__
	# pdb.set_trace()
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


def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	
	print('Total number of parameters: %d' % num_params)



def define_model(pretrained=False, model_root=None, encoder_model='Conv64F', classifier_model='DN4', norm='batch', init_type='normal', use_gpu=True, **kwargs):
	model = None
	norm_layer = get_norm_layer(norm_type=norm)

	if use_gpu:
		assert(torch.cuda.is_available())

	if classifier_model in ['ProtoNet', 'DN4']:
		model = Fewshot_model(encoder_model=encoder_model, classifier_model=classifier_model, **kwargs)

	else:
		raise NotImplementedError('Model name [%s] is not recognized' % classifier_model)
	
	# init_weights(model, init_type=init_type)
	print_network(model)

	if use_gpu:
		model.cuda()

	if pretrained:
		model.load_state_dict(model_root)

	return model




class Fewshot_model(nn.Module):
	'''
		Define a few-shot learning model, which consists of an embedding module and a classifier moduel.
	'''
	def __init__(self, encoder_model='Conv64F', classifier_model='DN4', class_num=64, way_num=5, shot_num=5, query_num=10, neighbor_k=3):
		super(Fewshot_model, self).__init__()
		self.encoder_model = encoder_model
		self.classifier_model = classifier_model
		self.way_num = way_num
		self.shot_num = shot_num
		self.query_num = query_num
		self.neighbor_k = neighbor_k
		self.loss_type = 'softmax'

		if   encoder_model == 'Conv64F':
			self.feature_dim = 64
		elif encoder_model == 'Conv64F_Local':
			self.feature_dim = 64
		elif encoder_model in ['ResNet10', 'ResNet18', 'ResNet34']:
			self.feature_dim = 512
		elif encoder_model in ['ResNet12', 'SeResNet12']:
			self.feature_dim = 640
		elif encoder_model in ['ResNet50', 'ResNet101']:
			self.feature_dim = 2048
		
		encoder_module    = encoder_dict[self.encoder_model]
		classifier_module = classifier_dict[self.classifier_model]

		self.features   = encoder_module()
		self.classifier = classifier_module(way_num=self.way_num, shot_num=self.shot_num, neighbor_k=self.neighbor_k)
	

		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				init.normal_(m.weight.data, 0.0, 0.02)
			elif isinstance(m, nn.Linear):
				init.normal_(m.weight.data, 0.0, 0.02)
			elif isinstance(m, nn.BatchNorm2d):
				init.normal_(m.weight.data, 1.0, 0.02)
				init.constant_(m.bias.data, 0.0)


	def forward(self, input1, input2, is_feature=False):
		
		# pdb.set_trace()
		x1 = self.features(input1)      # query:       75 * 64 * 21 * 21   
		x2 = self.features(input2)      # support set: 25 * 64 * 21 * 21  
		
		out = self.classifier(x1, x2)

		if is_feature:
			return x1, x2, out
		else:
			return out


class Model_with_reused_Encoder(nn.Module):
	'''
		Construct a new few-shot model by reusing a pre-trained embedding module.
	'''
	def __init__(self, pre_trained_model, new_classifier='DN4', way_num=5, shot_num=5, neighbor_k=3):
		super(Model_with_reused_Encoder, self).__init__()
		self.way_num = way_num
		self.shot_num = shot_num
		self.neighbor_k = neighbor_k
		self.model = pre_trained_model

		# Only use the features module
		self.features = nn.Sequential(
			*list(self.model.features.children())
			)

		classifier_module = classifier_dict[new_classifier]
		self.classifier = classifier_module(way_num=self.way_num, shot_num=self.shot_num, neighbor_k=self.neighbor_k)


	def forward(self, input1, input2):
		
		# pdb.set_trace()
		x1 = self.features(input1)
		x2 = self.features(input2)
		out = self.classifier(x1, x2)

		return out

