# Image-Captioning-Reproduction

Final Project for [CSC413/2516](https://fas.calendar.utoronto.ca/course/csc413h1) [(Winter 2021)](https://csc413-uoft.github.io/2021/).

Also, for CNN-LSTM, we modified and retrained [`muggins`'s implementation](https://github.com/muggin/show-and-tell). You can find our fork [here](https://github.com/rusbridger/show-and-tell).

### Members

- Shayan Khalili-Moghaddam
  - Contributions: CNN-LSTM network code and training, paper sections 3, 4(.0).
- Jiyu Nam
  - Contributions: kNN code and experiments; report sections 1, 2, 4, 4.1.
- Robert Wu
  - Contributions: CNN-LSTM network design/code; report sections 3, 4.2(.1), 4.3, Broader Impact.

## Abstract

With the boom of deep CNN models in the image recognition space in the last decade, as well as the increasingly accurate machine translation models using RNNs, the problem of generating novel image descriptions has experienced resurgence as an important intersection of the computer vision and natural language processing fields. In particular, the task of image caption retrieval can help us gain perspective as to how well these new caption generation models actually perform from a qualitative, or human, standpoint.

In this project, we implement the image description retrieval approach to the Image Captioning task, using a nearest neighbors algorithm (based on [this paper](https://arxiv.org/pdf/1505.04467.pdf)) to come up with a consensus for the best caption of an image.

## Nearest-Neigbors

### Generating a new Nearest-Neighbor graph

1. Download and save [MS COCO dataset](https://cocodataset.org/#download) in the root "data" and "annotations" folder
2. Fill in appropriate values for the variables `train_ann_file`, `valid_ann_file`, `k`, `train_early_stop`, `load_knn`, `res_file`, and `out_file` in "nearest-neighbors/evaluate.py"
3. Run `python evaluate.py` from within the "nearest-neighbors" directory
4. The nearest-neighbor graph is picked and saved in the "nearest-neighbors/knn-models" and can be provided as `load_knn` in the "nearest-neighbors/evaluate.py" file.

### Getting captions for images

To get prediction for random 5 images from validation set:

1. Ensure train and validation annotation files in the "annotations" folder (optionally, edit "nearest-neighbors/run.py where indicated).
2. Run `python run.py [path/to/knn-model]` from within the "nearest-neighbors" directory where the name of the knn-model file is of the form "knn_k={k}\_num\_{train_early_stop}".
3. Results are printed to console, alongside a sample of the real caption and the 4-gram BLEU scores.

To get prediction for user defined image:

1. Ensure train and validation annotation files in the "annotations" folder (optionally, edit "nearest-neighbors/run.py where indicated).
2. Run `python run.py [path/to/knn-model] [path/to/user/image]` from within the "nearest-neighbors" directory where the name of the knn-model file is of the form "knn*k={k}\_num*{train_early_stop}".
3. Results are printed to console as well as saved on the "nearest-neighbors/results" folder

## Requirements

- numpy
- pandas
- nltk
- PIL Image
- torch, torchvision
- sklearn
- pycocotools
- pycocoevalcap
- requests
- tqdm
