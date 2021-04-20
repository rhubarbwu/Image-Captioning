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
    def __init__(self):
        """ ImageFeature
        
        """

        # Load pretrained model and select layer
        self.model = models.resnet50(pretrained=True)
        self.layer = self.model._modules.get('avgpool')
        self.model.eval()

        # ResNet-50 expects images to be at least 224x224 and normalized to a specific mean and std dev
        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()


    def get_vector(self, img):
        """
        :param img (PIL Image or str): image (or path to img) to extract features from
        """
        # Read image and save transformed image
        if type(img) == str:
            if img.startswith('http'):
              img = Image.open(requests.get(img, stream=True).raw)
            else:
              img = Image.open(img)

        self.img = Variable(self.normalize(self.to_tensor(self.scaler(img.convert("RGB")))).unsqueeze(0))
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
