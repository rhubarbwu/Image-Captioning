# Image-Captioning-Reproduction

Final Project for [CSC413/2516](https://fas.calendar.utoronto.ca/course/csc413h1) [(Winter 2021)](https://csc413-uoft.github.io/2021/).

### Members

- Shayan Khalili-Moghaddam
- Jiyu Nam
- Robert Wu

## Abstract

With the boom of deep CNN models in the image recognition space in the last decade, as well as the increasingly accurate machine translation models using RNNs, the problem of generating novel image descriptions has experienced resurgence as an important intersection of the computer vision and natural language processing fields. In particular, the task of image caption retrieval can help us gain perspective as to how well these new caption generation models actually perform from a qualitative, or human, standpoint.

In this project, we propose comparing the approach of generating novel image descriptions using a deep image CNNs to encode image features, and an LSTM network for decoding those features into sentences (based on the ["Show and Tell" model from Google](https://arxiv.org/pdf/1411.4555.pdf)), with the approach of image description retrieval using a nearest neighbors algorithm (based on [this paper](https://arxiv.org/pdf/1505.04467.pdf)). to come up with a consensus for the best caption of an image. This will likely require building extensive dictionary or phrase indexes of text from, mostly from our chosen datasets.

## Nearest Neigbours

## Recurrent Neural Networks (RNN)

## To-Do

1. Data preprocessing for network training loop.
2. ~~Encoder-Decoder architecture.~~
3. Implement training loop.
4. Use a CNN pre-trained on ImageNet and train in two ways to compare.
   - Train Decoder and last layer of Encoder on MS COCO.
   - Train Deocder and fine-tune entire Encoder on MS COCO.
5. Explore different resizing algorithms in the DataLoader.
6. Assess similarity of different captions produced by CNN-LSTM.
   - Sample words until end token or max token length is reached.
   - BeamSearch: "iteratively consider the set of `k` best sentences up to time $t$ as candidates to generate sentences of size `t+1`, and keep only the resulting best `k` of them. [Show and Tell, page 4](https://arxiv.org/pdf/1411.4555.pdf)
7. Compare this RNN-CNN architecture to Nearest Neighbours.
