"""
Extract feature vector from the "avgpool" layer of ResNet-50 trained off of the ImageNet dataset

Based off the work of Christina Safka (https://github.com/christiansafka/img2vec/)
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import requests

class ImageFeature():
	def __init__(self, img_path):
		""" ImageFeature
		:param img_path (str): path to image to extract features (can be url beginning with http or local)
		"""

		# Load pretrained model and select layer
		self.model = models.resnet50(pretrained=True)
		self.layer = self.model._modules.get('avgpool')
		self.model.eval()

		# ResNet-18 expects images to be at least 224x224 and normalized to a specific mean and std dev
		scaler = transforms.Resize((224, 224))
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
										 std=[0.229, 0.224, 0.225])
		to_tensor = transforms.ToTensor()

		# Read image and save transformed image
		if img_path.startswith('http'):
			raw_img = Image.open(requests.get(img_path, stream=True).raw)
		else:
			raw_img = Image.open(img_path)
		self.img = Variable(normalize(to_tensor(scaler(raw_img.convert("RGB")))).unsqueeze(0))


	def get_vector(self):
	    my_embedding = torch.zeros(2048) # 'avgpool' layer has an output size of 2048

	    # Define a function that will copy the output of a layer and attach to selected layer
	    def copy_data(m, i, o):
        	my_embedding.copy_(o.data.reshape(o.data.size(1)))
	    h = self.layer.register_forward_hook(copy_data)

	    # Run model
	    self.model(self.img)

	    # Detach our copy function from the layer
	    h.remove()

	    return my_embedding
