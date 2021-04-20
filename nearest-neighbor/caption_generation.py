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
from pycocoevalcap.bleu.bleu import Bleu
import pickle

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
        :param imgs (list of PIL Images or str): PIL images (or img paths) to generate captions for
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
            # get best caption based on highest BLEU score
            consensus_caption = self.consensus_caption(cluster)
            best_captions.append(consensus_caption)

        return best_captions

    def consensus_caption(self, cluster):
        """
        Given a cluster of img_ids, find consensus caption
        """
        # get a set of all captions in the cluster
        k = len(cluster)
        # all_raw_captions = [None] * (5*k)
        # indices = np.where(np.in1d(self.coco.ids,cluster))[0]
        ann_ids = self.coco.coco.getAnnIds(cluster)
        all_raw_captions = self.coco.coco.loadAnns(ann_ids)
        all_raw_captions = [caption_dict['caption'] for caption_dict in all_raw_captions]
        all_raw_captions = np.array(all_raw_captions, dtype=object)
        # for i, idx in enumerate(indices):
            # raw_captions = self.coco[idx][-1]
            # all_raw_captions[i*5:(i+1)*5+1] = raw_captions

        # calculate BLEU score for each caption against its cluster
        caption_scores = np.zeros(len(all_raw_captions)) 
        for i, hypothesis in enumerate(all_raw_captions):
            # mask = np.ones(len(all_raw_captions), dtype=bool)
            # mask[i] = False
            # references = all_raw_captions[mask]
            references = np.delete(all_raw_captions, i)
            scores, _ = Bleu(4).compute_score({i:list(references)},{i:[hypothesis]}, verbose=0)
            caption_scores[i] = scores[3] # 4-gram

        # for i in range(5*k):
            # references = all_raw_captions.copy()
            # hypothesis = [references.pop(i)]
            # scores, _ = Bleu(4).compute_score({i:references},{i:hypothesis}, verbose=0)
            # caption_scores[i] = scores[3] # 4-gram

        return all_raw_captions[caption_scores.argmax()]

    def evaluate(self, dataset, early_stop=None):
        """
        function used to evaluate caption generation
        :param dataset (torchvision.datasets.COCOCaptions): validation dataset
        """
        num_imgs = early_stop if early_stop else len(dataset)

        print("Now building nearest neighbor graph...")
        img_feats = np.empty((num_imgs, 2048))
        cap_map = []
        idx = -1
        for img, caps in tqdm(dataset):
            img, caps = dataset[idx]
            img_id = dataset.ids[idx]
            img_feat = self.img_feature_obj.get_vector(img)
            img_feats[idx] = img_feat.numpy()
            cap_map.append({"image_id":img_id, "caption":caps})
            idx += 1
            break
            if idx == num_imgs:
                break

        self.nearest_neighbors = self.get_kneighbors(img_feats)

        print("Now getting best captions...")
        best_captions = [None] * len(self.nearest_neighbors)
        idx = 0
        for cluster in tqdm(self.nearest_neighbors):
            img_id = dataset.ids[idx]
            consensus_caption = self.consensus_caption(cluster)
            best_captions[idx] = {"image_id": img_id, "caption":consensus_caption}
            idx += 1

        return best_captions, cap_map
            


