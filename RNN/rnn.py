from pycocotools.coco import COCO

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CocoCaptions

from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

from typing import Any, Callable, Optional, Tuple, List

import time

coco = COCO("../data/annotations/captions_train2014.json")

tokenizer = get_tokenizer('basic_english')
counter = Counter()
for img_id in tqdm(list(coco.imgs.keys())):
    annsIds = coco.getAnnIds(imgIds=[img_id])
    caption = coco.loadAnns(annsIds[0])[0]['caption']
    counter.update(tokenizer(caption))
vocab = Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

bos_index = vocab['<bos>']
eos_index = vocab['<eos>']
pad_index = vocab['<pad>']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_batch(data_batch):
    img_batch, cap_batch = [], []
    for (img, cap) in data_batch:
        img_batch.append(img.unsqueeze(0))
        cap_batch.append(cap)
    
    img_batch = torch.cat(img_batch, dim=0)
    cap_batch = pad_sequence(cap_batch, padding_value=pad_index)
    
    return img_batch, cap_batch

def sample_caption(encoder, decoder, img, max_len=20):
    hs = decoder.hidden_size
    
    greedy_output = torch.zeros(max_len)
    features = encoder(img.unsqueeze(0))
    
    output = torch.full((1, 1), bos_index).to(device)
    
    for t in range(1, max_len):
        output, (hidden, cell) = decoder(output, hidden, cell) # output: (1, B, V), hidden: (1, B, H), cell: (1, B, H)
        output = output.argmax(dim=2) # top1: (1, 1)
        greedy_output[t] = output
    
    return greedy_output

def sample_captions(encoder, decoder, img_batch, max_len=20):
    bs = img_batch.shape[0]
    hs = decoder.hidden_size
    
    outputs = torch.zeros(max_len, bs, len(vocab)).to(device) # outputs: (S, B, V)
    features = encoder(img_batch)
    features = features.view(1, features.shape[0], features.shape[1])
    hidden = features
    cell = features
    
    output = torch.full((1, bs), bos_index).to(device) # output: (1, B)
    
    for t in range(1, max_len):
        output, (hidden, cell) = decoder(output, hidden, cell) # output: (1, B, V), hidden: (1, B, H), cell: (1, B, H)
        outputs[t] = output
        
        output = output.argmax(dim=2) # output: (1, B)
    
    return outputs

def train(train_loader, val_loader, encoder, decoder, critereon, optimizer, train_losses, val_losses):
    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        
        start_time = time.time()
        
        for images, captions in tqdm(train_loader):
            images, captions = images.to(device), captions.to(device)
            
            optimizer.zero_grad()
            features = encoder(images)
            features = features.view(1, features.shape[0], features.shape[1])
            output, _ = decoder(captions, features, features)

            output = output[1:].view(-1, output.shape[-1])
            captions = captions[1:].view(-1)
            
            loss = critereon(output, captions)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        for images, captions in tqdm(val_loader):
            images, captions = images.to(device), captions.to(device)
            
            output = sample_captions(encoder, decoder, images, max_len=captions.shape[0])

            output = output[1:].view(-1, output.shape[-1])
            captions = captions[1:].view(-1)
            
            loss = critereon(output, captions)
            
            val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        print('Epoch: {} | Time: {}s'.format(epoch+1, elapsed_time))
        print('\tTrain Loss: {}'.format(train_loss))
        print('\tVal Loss: {}'.format(val_loss))

class Encoder(nn.Module):
    
    def __init__(self, hidden_size):
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        
        resnet = models.resnet50(pretrained=True)
        
        for params in resnet.parameters(): # will not be fine-tuning resnet
            params.requires_grad = False
        
        self.resnet = nn.Sequential(*(list(resnet.children())[:-1])) # will not be using last layer of resnet since that layer outputs a 1000-D vector for imagenet classification
        self.embedding = nn.Linear(2048, hidden_size) # add a linear layer to get a feature vector to pass into the decoder
        
    def forward(self, inputs):
        output = self.resnet(inputs)
        output = output.view(output.size(0), -1)
        output = self.embedding(output)
        return output
    
def tokenize_caption(cap):
    return torch.cat([
        torch.tensor([bos_index], dtype=torch.long),
        torch.tensor([vocab[token] for token in tokenizer(cap[0])], dtype=torch.long),
        torch.tensor([eos_index], dtype=torch.long)
    ])
    
class Decoder(nn.Module):
    
    def __init__(self, vocab_size, hidden_size):
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size) # word embeddings will be size hidden_size, hidden states will be size hidden_size
        self.out = nn.Linear(in_features=hidden_size, out_features=vocab_size)
    
    def forward(self, inputs, h_0, c_0):
        """
        inputs: (S, B) set of captions, each entry is an integer index into the vocabulary
        h_0: (1, B, H) initial hidden state
        c_0: (1, B, H) initial cell state
        """
        input_embedded = self.embedding(inputs) # input_embedded: (S, B, H)
        output, (hidden, cell) = self.rnn(input_embedded, (h_0, c_0)) # output: (S, B, H), hidden: (1, B, H), cell: (1, B, H)
        output = self.out(output) # output: (S, B, V)
        return output, (hidden, cell)

if __name__ == '__main__':
    
    val_dataset_untransformed = CocoCaptions(
        root="../data/images/val2014", 
        annFile="../data/annotations/captions_val2014.json",
    )
    
    batch_size = 128
    learning_rate = 0.005
    num_epochs = 10
    hidden_size = 128
    
    train_dataset = CocoCaptions(
    root="../data/images/train2014", 
    annFile="../data/annotations/captions_train2014.json", 
    transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
    target_transform=tokenize_caption
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=generate_batch,
        num_workers=4
    )

    val_dataset = CocoCaptions(
    root="../data/images/val2014", 
    annFile="../data/annotations/captions_val2014.json", 
    transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
    target_transform=tokenize_caption
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=generate_batch,
        num_workers=4
    )

    encoder = Encoder(hidden_size).to(device)
    decoder = Decoder(len(vocab), hidden_size).to(device)

    critereon = nn.CrossEntropyLoss(ignore_index=pad_index)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    
    train_losses, val_losses = [], []
    train(train_loader, val_loader, encoder, decoder, critereon, optimizer, train_losses, val_losses)