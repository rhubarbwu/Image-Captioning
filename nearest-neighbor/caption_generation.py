"""
Generates caption based on K-NearestNeighbour approach

Based off the work by kayburns (https://github.com/kayburns/img-captioning-baseline)
"""

from PIL import Image
import requests
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from pycocotools.coco import COCO
# from nltk.translate import bleu_score
from pycocoevalcap.bleu.bleu import Bleu
import pickle
import numba

from feature_extraction import ImageFeature

class CaptionGenerator():
    def __init__(self, coco_dataset, k=10, early_stop=None, load_knn=None):
        """ CaptionGenerator
        :param coco_dataset (torchvision.datasets.COCOCaptions): COCOCaptions object with image directory and annotation path
        :param k (int): number of clusters
        :param early_stop (int): optional parameter of how many images to use from dataset
        :param load_knn (str): optional parameter of path for NearestNeighbor object to load and use
        """

        self.coco = coco_dataset
        self.img_feature_obj = ImageFeature()

        if load_knn:
            self.neigh = pickle.load(open(load_knn, 'rb'))
            self.img_map = pickle.load(open(load_knn+"img_map", 'rb'))
        else:
            # Extract image features using `feature_extraction` for all images from loaded coco object
            self.num_imgs = len(self.coco.ids)
            if early_stop:
                self.num_imgs = early_stop
                print(f"Using early stop at {self.num_imgs} images")

            self.extract_features()

            # Fit entire dataset to k nearest neighbours
            self.neigh = NearestNeighbors(n_neighbors=k, metric='cosine')
            self.neigh = self.neigh.fit(self.img_feats)

            # Save KNN model and img map
            file_path = f'./knn-models/knn_k={k}_num_{self.num_imgs}'
            neighPickle = open(file_path, 'wb')
            pickle.dump(self.neigh, neighPickle)  
            mapPickle = open(file_path+"img_map", "wb")
            pickle.dump(self.img_map, mapPickle)

    @numba.jit(fastmath=True)
    def extract_features(self):
        self.img_feats = np.empty((self.num_imgs, 2048))
        self.img_map = {}
        idx = 0
        for img, captions in tqdm(self.coco):
            img_id = self.coco.ids[idx]
            img_feat = self.img_feature_obj.get_vector(img)
            self.img_feats[idx] = img_feat.numpy()
            self.img_map[idx] = img_id
            idx += 1
            if idx == self.num_imgs:
                break

    def get_kneighbors(self, img_feats):
        """
        :param img_feats (np.array): features of the images we would like the k nearest neighbors for
        """
        nearest_neighbors = self.neigh.kneighbors(img_feats, return_distance=False)
        nearest_neighbors = np.vectorize(self.img_map.get)(nearest_neighbors) # convert idx ids to coco img ids
        return nearest_neighbors

    def get_captions(self, imgs):
        """
        :param imgs (list of PIL Images): PIL images to generate captions for
        """

        # Extract features for images and get k neighbors
        img_feats = np.empty((len(imgs), 2048))
        idx = 0
        for img in imgs:
            img_feat = self.img_feature_obj.get_vector(img)
            img_feats[idx] = img_feat.numpy()
            idx += 1

        self.nearest_neighbors = self.get_kneighbors(img_feats)

        # Find consensus caption -- for each caption in cluster, calculate BLEU score from within cluster. Return highest BLEU score caption
        best_captions = []
        for idx, cluster in enumerate(self.nearest_neighbors):
            # get a set of all captions in the cluster
            all_cap_ids = [self.coco.coco.getAnnIds(img_id) for img_id in cluster]
            all_cap_ids = [cap_id for cap_ids in all_cap_ids for cap_id in cap_ids]
            raw_captions = [self.coco.coco.anns[cap_id]['caption'] for cap_id in all_cap_ids]
            # all_captions = [self.coco.coco.anns[cap_id]['caption'].split() for cap_id in all_cap_ids] # get list of 5*k captions

            # calculate BLEU score for each caption against its cluster
            caption_scores = np.zeros(len(raw_captions)) 
            for i in range(len(raw_captions)):
                references = raw_captions.copy()
                hypothesis = [references.pop(i)]
                scores, _ = Bleu().compute_score({idx:references},{idx:hypothesis})
                caption_scores[i] = scores[2] # 3-gram
            # get best caption based on highest BLEU score
            best_caption = raw_captions[caption_scores.argmax()]
            best_captions.append(best_caption)

        return best_captions
