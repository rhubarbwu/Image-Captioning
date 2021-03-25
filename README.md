# Image-Captioning-Reproduction
CSC413 Final Project 2021

## About the Project
With the boom of deep CNN models in the image recognition space in the last decade, as well as the increasingly accurate machine translation models using RNNs, the problem of generating novel image descriptions has experienced resurgence as an important intersection of the computer vision and natural language processing fields. In particular, the task of image caption retrieval can help us gain perspective as to how well these new caption generation models actually perform from a qualitative, or human, standpoint.

In this project, we propose comparing the approach of generating novel image descriptions using a deep image CNNs to encode image features, and an LSTM network for decoding those features into sentences (based on the ["Show and Tell" model from Google](https://arxiv.org/pdf/1411.4555.pdf)), with the approach of image description retrieval using a nearest neighbors algorithm (based on [this paper](https://arxiv.org/pdf/1505.04467.pdf)). to come up with a consensus for the best caption of an image. This will likely require building extensive dictionary or phrase indexes of text from, mostly from our chosen datasets.
