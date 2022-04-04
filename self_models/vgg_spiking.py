#---------------------------------------------------
# Imports
#---------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
from collections import OrderedDict
from matplotlib import pyplot as plt
import copy

# cfg: pre-computed CNN architechture
cfg = {
    'VGG5' : [64, 'A', 128, 128, 'A'],
    'VGG9':  [64, 'A', 128, 256, 'A', 256, 512, 'A', 512, 'A', 512],
	'VGG11': [64, 'A', 128, 256, 'A', 512, 512, 'A', 512, 'A', 512, 512], # 'VGG11': [64, 'A', 128, 256, 256, 'A', 512, 512, 512, 'A', 512, 512],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512],
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512]
}
class PoissonGenerator(nn.Module):
	
	def __init__(self):
		super().__init__()

	def forward(self,input):
		
		out = torch.mul(torch.le(torch.rand_like(input), torch.abs(input)*1.0).float(), torch.sign(input))
		return out

class STDB(torch.autograd.Function):

	alpha 	= ''
	beta 	= ''
    
	@staticmethod
	def forward(ctx, input, last_spike):
        
		ctx.save_for_backward(last_spike)
		out = torch.zeros_like(input).cuda()
		out[input > 0] = 1.0
		return out

	@staticmethod
	def backward(ctx, grad_output):
	    		
		last_spike, = ctx.saved_tensors
		grad_input = grad_output.clone()
		grad = STDB.alpha * torch.exp(-1*last_spike)**STDB.beta
		return grad*grad_input, None

class LinearSpike(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    gamma = 0.3 # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input, last_spike):
        
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,     = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad       = LinearSpike.gamma*F.threshold(1.0-torch.abs(input), 0, 0)
        return grad*grad_input, None

class VGG_SNN_STDB(nn.Module):

	def __init__(self, vgg_name, activation='Linear', labels=10, timesteps=100, leak=1.0, default_threshold = 1.0, alpha=0.3, beta=0.01, dropout=0.2, kernel_size=3, dataset='CIFAR10'):
		super().__init__()
		
		self.vgg_name 		= vgg_name
		if activation == 'Linear':
			self.act_func 	= LinearSpike.apply
		elif activation == 'STDB':
			self.act_func	= STDB.apply
		self.labels 		= labels
		self.timesteps 		= timesteps
		self.leak 	 		= torch.tensor(leak)
		STDB.alpha 		 	= alpha
		STDB.beta 			= beta 
		self.dropout 		= dropout
		self.kernel_size 	= kernel_size
		self.dataset 		= dataset
		self.input_layer 	= PoissonGenerator()
		self.threshold 		= {}
		self.mem 			= {}
		self.mask 			= {}
		self.spike 			= {}
		self.count_spike = {}
		self.count_num = 0
		
		self.features, self.classifier = self._make_layers(cfg[self.vgg_name]) # features--CNN structure; classifier--ANN structure
		
		self._initialize_weights2()

		for l in range(len(self.features)):
			if isinstance(self.features[l], nn.Conv2d):
				self.threshold[l] 	= torch.tensor(default_threshold)
				
		prev = len(self.features)
		for l in range(len(self.classifier)-1):
			if isinstance(self.classifier[l], nn.Linear):
				self.threshold[prev+l] 	= torch.tensor(default_threshold)

	def _initialize_weights2(self):
		for m in self.modules():
            
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				if m.bias is not None:
					m.bias.data.zero_()

	def threshold_update(self, scaling_factor=1.0, thresholds=[]):

		# Initialize thresholds
		self.scaling_factor = scaling_factor
		
		for pos in range(len(self.features)):
			if isinstance(self.features[pos], nn.Conv2d):
				if thresholds:
					self.threshold[pos] = torch.tensor(thresholds.pop(0)*self.scaling_factor)
		
		prev = len(self.features)

		for pos in range(len(self.classifier)-1):
			if isinstance(self.classifier[pos], nn.Linear):
				if thresholds:
					self.threshold[prev+pos] = torch.tensor(thresholds.pop(0)*self.scaling_factor)
				

	def _make_layers(self, cfg):
		layers 		= []
		if self.dataset =='MNIST':
			in_channels = 1
		else:
			in_channels = 3

		for x in (cfg):
			stride = 1
						
			if x == 'A':
				layers.pop()
				layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
			
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=stride, bias=False),
							nn.ReLU(inplace=True)
							]
				layers += [nn.Dropout(self.dropout)]
				in_channels = x
		
		features = nn.Sequential(*layers)
		
		layers = []
		if self.vgg_name == 'VGG11': #and self.dataset=='CIFAR100':
			layers += [nn.Linear(512*2*2, 1024*4, bias=False)] # layers += [nn.Linear(512*4*4, 1024, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(1024*4, 1024*4, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(1024*4, self.labels, bias=False)]

		elif self.vgg_name == 'VGG5' and self.dataset != 'MNIST':
			layers += [nn.Linear(512*4*4, 1024*4, bias=False)] # default = 512*4*4, 1024
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(1024*4, 1024*4, bias=False)] # default = 1024, 1024
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(1024*4, self.labels, bias=False)]
		
		elif self.vgg_name != 'VGG5' and self.dataset != 'MNIST':
			layers += [nn.Linear(512*2*2, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(4096, self.labels, bias=False)]
		
		elif self.vgg_name == 'VGG5' and self.dataset == 'MNIST':
			layers += [nn.Linear(128*7*7, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(4096, self.labels, bias=False)]

		elif self.vgg_name != 'VGG5' and self.dataset == 'MNIST':
			layers += [nn.Linear(512*1*1, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(4096, self.labels, bias=False)]


		classifer = nn.Sequential(*layers)
		return (features, classifer)

	def network_update(self, timesteps, leak):
		self.timesteps 	= timesteps
		self.leak 	 	= torch.tensor(leak)
	
	def neuron_init(self, x):
		self.batch_size = x.size(0)
		self.width 		= x.size(2)
		self.height 	= x.size(3)

		for l in range(len(self.features)):
								
			if isinstance(self.features[l], nn.Conv2d):
				self.mem[l] 		= torch.zeros(self.batch_size, self.features[l].out_channels, self.width, self.height)
				if self.count_num == 0:
					self.count_spike[l] = torch.zeros(self.features[l].in_channels * self.kernel_size*self.kernel_size)
			elif isinstance(self.features[l], nn.Dropout):
				self.mask[l] = self.features[l](torch.ones(self.mem[l-2].shape))

			elif isinstance(self.features[l], nn.AvgPool2d):
				self.width = self.width//self.features[l].kernel_size
				self.height = self.height//self.features[l].kernel_size
		
		prev = len(self.features)

		for l in range(len(self.classifier)):
			
			if isinstance(self.classifier[l], nn.Linear):
				self.mem[prev+l] 		= torch.zeros(self.batch_size, self.classifier[l].out_features)
				if self.count_num == 0:
					self.count_spike[l+prev] = torch.zeros(self.classifier[l].in_features)
			
			elif isinstance(self.classifier[l], nn.Dropout):
				self.mask[prev+l] = self.classifier[l](torch.ones(self.mem[prev+l-2].shape))
				
		self.spike = copy.deepcopy(self.mem)
		if self.count_spike == {}:
			self.count_spike = copy.deepcopy(self.mem)
		for key, values in self.spike.items():
			for value in values:
				value.fill_(-1000)

	def forward(self, x, find_max_mem=False, max_mem_layer=0): # input data: x (batch_size, 3, 32, 32)
		
		self.neuron_init(x)
		max_mem=0.0
		
		for t in range(self.timesteps): # forward for every timesteps
			out_prev = self.input_layer(x) # PoissonGenerator--rate coding
			
			for l in range(len(self.features)): # the previous CNN structure-like Conv2d Avgpool
				
				if isinstance(self.features[l], (nn.Conv2d)): # figure out the Lth layer.
					
					if find_max_mem and l == max_mem_layer:
						if (self.features[l](out_prev)).max() > max_mem:
							max_mem = (self.features[l](out_prev)).max()
						break

					mem_thr 		= (self.mem[l]/self.threshold[l]) - 1.0 # get the threshold, self.mem for all the conv2 and FC layer
					out 			= self.act_func(mem_thr, (t-1-self.spike[l])) # save the ((t-1-self.spike[l])) and out: (mem_thr > 0) = 1
					rst 			= self.threshold[l]* (mem_thr>0).float()
					self.spike[l] 	= self.spike[l].masked_fill(out.bool(),t-1) # 'True' elements of out.bool() in self.spike[l] will be replaced by t-1
					# if self.batch_size == 64:
					# 	self.count_spike[l] += out
					self.mem[l] 	= self.leak*self.mem[l] + self.features[l](out_prev) - rst # LIF

					# ********** count spike *****************
					n1, n2, n3, n4 = out_prev.shape
					for xx in range(n1):
						for yy in range(n3):
							yy -= 1
							for zz in range(n4):
								zz -= 1
								tmp_const = n3 - 2
								tmp_spike = torch.zeros((n2, 3, 3))
								if yy == -1 and zz == -1: # top left
									tmp_spike[:, 1:3, 1:3] = out_prev[xx, :, 1:3, 1:3]
								elif yy == tmp_const and zz == tmp_const: # bottom right
									tmp_spike[:, 0:2, 0:2] = out_prev[xx, :, tmp_const:tmp_const+2, tmp_const:tmp_const+2]
								elif yy == -1 and zz != tmp_const: # top
									tmp_spike[:, 1:3, :] = out_prev[xx, :, 1:3, zz:zz+3]
								elif yy != tmp_const and zz == -1: # left
									tmp_spike[:, :, 1:3] = out_prev[xx, :, yy:yy+3, 1:3]
								elif yy == -1 and zz == tmp_const: # top right
									tmp_spike[:, 1:3, 0:2] = out_prev[xx, :, 1:3, tmp_const:tmp_const+2]
								elif yy == tmp_const and zz == -1: # bottom left
									tmp_spike[:, 0:2, 1:3] = out_prev[xx, :, tmp_const:tmp_const+2, 1:3]
								elif yy == tmp_const and zz != -1: # bottom
									tmp_spike[:, 0:2, :] = out_prev[xx, :, tmp_const:tmp_const+2, zz:zz + 3]
								elif yy != -1 and zz == tmp_const: # right
									tmp_spike[:, :, 0:2] = out_prev[xx, :, yy:yy + 3, tmp_const:tmp_const+2]

								# self.count_spike[l] += out_prev[xx, :, yy:yy + 3, zz:zz + 3].reshape(n2*3*3) * 1e-5 # record the input
								self.count_spike[l] += tmp_spike.reshape(n2*3*3) * 1e-5 # record the input
					# *************************************

						# else:
						# 	for yy in range(n3*n3): # only for VGG
						# 		self.count_spike[l] += out_prev[xx, :, yy:yy + 3, yy:yy + 3].reshape(n2*3*3) * 1e-5
					out_prev  		= out.clone() # python share the address, therefore need to use clone to allocate a new space
					del out
					del mem_thr
					del rst
					self.count_num += 1e-5

				elif isinstance(self.features[l], nn.AvgPool2d):
					out_prev 		= self.features[l](out_prev)
				
				elif isinstance(self.features[l], nn.Dropout):
					out_prev 		= out_prev * self.mask[l] # self.mask for all the dropout layer and the drop will be 1 at the inference stage
			
			if find_max_mem and max_mem_layer<len(self.features):
				continue

			out_prev       	= out_prev.reshape(self.batch_size, -1)
			prev = len(self.features)
			
			for l in range(len(self.classifier)-1):
													
				if isinstance(self.classifier[l], (nn.Linear)):
					
					if find_max_mem and (prev+l)==max_mem_layer:
						if (self.classifier[l](out_prev)).max()>max_mem:
							max_mem = (self.classifier[l](out_prev)).max()
						break

					mem_thr 			= (self.mem[prev+l]/self.threshold[prev+l]) - 1.0
					out 				= self.act_func(mem_thr, (t-1-self.spike[prev+l]))
					rst 				= self.threshold[prev+l] * (mem_thr>0).float()
					self.spike[prev+l] 	= self.spike[prev+l].masked_fill(out.bool(),t-1)
					# if self.batch_size == 64:
					# 	self.count_spike[prev+l] += out
					self.mem[prev+l] 	= self.leak*self.mem[prev+l] + self.classifier[l](out_prev) - rst

					# ***************** count spike ***************
					n1, n2 = out_prev.shape
					for xx in range(n1):
						self.count_spike[l+prev] += out_prev[xx, :] * 1e-5
					# *********************************************

					out_prev  		= out.clone()
					del out
					del mem_thr
					del rst
					self.count_num += 1e-5
					# ************* count spike ***************
					if l == 5:
						for xx in range(n1):
							self.count_spike[6 + prev] += out_prev[xx, :] * 1e-5
					# *****************************************
				elif isinstance(self.classifier[l], nn.Dropout):
					out_prev 		= out_prev * self.mask[prev+l]
			# l += 1
			# for xx in range(n1):
			# 	self.count_spike[1 + l + prev] += out_prev[xx, :] * 1e-5
			# Compute the classification layer outputs
			if not find_max_mem:
				self.mem[prev+l+1] 		= self.mem[prev+l+1] + self.classifier[l+1](out_prev)
			del out_prev
		if find_max_mem:
			return max_mem

		# if self.count_num == 100:
		# 	for i in self.count_spike.keys():
		# 		name = 'Spike_test_input_1_' + str(i) + '.npy'
		# 		np.save(name, self.count_spike[i].cpu().numpy())
		# 	print(1)
		# if self.count_num == 1000:
		# 	for i in self.count_spike.keys():
		# 		name = 'Spike_test_input_10_' + str(i) + '.npy'
		# 		np.save(name, self.count_spike[i].cpu().numpy())
		# 	print(10)
		# if self.count_num == 10000:
		# 	for i in self.count_spike.keys():
		# 		name = 'Spike_test_input_100_' + str(i) + '.npy'
		# 		np.save(name, self.count_spike[i].cpu().numpy())
		# 	print(100)
		# if self.count_num % 10000 == 0:
		# 	print(self.count_num//100)
		return self.mem[prev+l+1]



