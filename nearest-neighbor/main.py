"""
main file to run everything
"""
from caption_generation import CaptionGenerator
from feature_extraction import ImageFeature

from PIL import Image
import matplotlib.pyplot as plt
import requests
import glob
import random
import warnings
warnings.filterwarnings("ignore")

def main():
    data_dir = '../annotations/annotations_trainval2014'
    data_type = 'val2014'
    ann_file = '{}/annotations/captions_{}.json'.format(data_dir, data_type)

    k = 10
    early_stop = 1000
    capgen = CaptionGenerator(ann_file, k=k, early_stop=early_stop)

    # all_imgs = glob.glob(f'../data/{data_type}/{data_type}/*.jpg')
    # sample_imgs = all_imgs[-5:]
    sample_img_urls = []
    sample_img_ids = []
    for i in range(5):
        img_id, img_info = random.choice(list(capgen.coco.imgs.items()))
        sample_img_urls.append(img_info['coco_url'])
        sample_img_ids.append(img_id)
    best_captions = capgen.get_captions(sample_img_urls)

    print("-------------------------------------RESULTS-------------------------------------")
    for idx, img_id in enumerate(sample_img_ids):
        print(f"ID: {img_id} \n Real caption: {capgen.coco.imgToAnns[img_id][0]['caption']} \n Sampled caption: {best_captions[idx]}")

if __name__ == "__main__":
    main()