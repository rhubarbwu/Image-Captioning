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
from nltk.translate import bleu_score
import pickle
import numba

from feature_extraction import ImageFeature

class CaptionGenerator():
	def __init__(self, ann_file, k=10, early_stop=None, load_knn=None):
		""" CaptionGenerator
		:param ann_file (str): path to annotation file to use from COCO dataset (train vs. val vs. test). 
							   Ex: '../annotations/annotations_trainval2014/annotations/captions_val2014.json'
		:param k (int): number of clusters
		:param early_stop (int): optional parameter of how many images to use from dataset
		:param load_knn (str): optional parameter of path for NearestNeighbor object to load and use
		"""

		# Load COCO data from annotation file
		self.coco = COCO(ann_file)

		if load_knn:
			self.neigh = pickle.load(open(load_knn, 'rb'))
			self.img_map = pickle.load(open(load_knn+"img_map", 'rb'))
		else:
			# Extract image features using `feature_extraction` for all images from loaded coco object
			self.num_imgs = len(self.coco.imgs)
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
		img_ids = list(self.coco.imgs.keys())[:self.num_imgs]
		for img_id in tqdm(img_ids):
		    url = self.coco.imgs[img_id]['coco_url']
		    img_feat = ImageFeature(url).get_vector()
		    self.img_feats[idx] = img_feat.numpy()
		    self.img_map[idx] = img_id
		    idx += 1

	def get_kneighbors(self, img_feats):
		"""
		:param img_feats (np.array): features of the images we would like the k nearest neighbors for
		"""
		nearest_neighbors = self.neigh.kneighbors(img_feats, return_distance=False)
		nearest_neighbors = np.vectorize(self.img_map.get)(nearest_neighbors) # convert idx ids to coco img ids
		return nearest_neighbors

	def get_captions(self, img_paths):
		"""
		:param img_paths (list of str): list of paths to images we want the captions for 
		"""

		# Extract features for images and get k neighbors
		img_feats = np.empty((len(img_paths), 2048))
		idx = 0
		for img_path in img_paths:
			img_feat = ImageFeature(img_path).get_vector()
			img_feats[idx] = img_feat.numpy()
			idx += 1

		self.nearest_neighbors = self.get_kneighbors(img_feats)

		# Find consensus caption -- for each caption in cluster, calculate BLEU score from within cluster. Return highest BLEU score caption
		best_captions = []
		for cluster in self.nearest_neighbors:
		    ann_ids = self.coco.getAnnIds(imgIds=cluster)
		    ann_ids = [ann_ids[i:i + 5] for i in range(0, len(ann_ids), 5)]
		    
		    # get a set of all captions in the cluster and split by whitespace
		    all_captions = []
		    for img_caption_ids in ann_ids:
		        for caption_id in img_caption_ids:
		            all_captions.append(self.coco.anns[caption_id]['caption'].split())

		    # calculate BLEU score for each caption against its cluster
		    caption_scores = np.zeros(len(all_captions)) 
		    for i in range(len(all_captions)):
		        references = all_captions.copy()
		        hypothesis = references.pop(i)
		        caption_scores[i] = bleu_score.sentence_bleu(references, hypothesis)
		    
		    # get best caption based on highest BLEU score
		    best_caption = self.coco.getAnnIds(imgIds=cluster)[caption_scores.argmax()]
		    best_captions.append(self.coco.anns[best_caption]['caption'])

		return best_captions
