"""
main file to run everything
"""
from caption_generation import CaptionGenerator
from feature_extraction import ImageFeature

from PIL import Image
import matplotlib.pyplot as plt
from nltk.translate import bleu_score
from pycocotools.coco import COCO
import requests
import glob
import random
import warnings
warnings.filterwarnings("ignore")

def main():
    data_dir = '../annotations/annotations_trainval2014'
    data_type = 'train2014'
    ann_file = '{}/annotations/captions_{}.json'.format(data_dir, data_type)

    k = 10
    early_stop = 2000
    # load_knn = "./knn-models/knn_k=10_num_2000"
    load_knn = None
    capgen = CaptionGenerator(ann_file, k=k, early_stop=early_stop, load_knn=load_knn)

    # get images from validation file
    data_type = 'val2014'
    ann_file = '{}/annotations/captions_{}.json'.format(data_dir, data_type)
    coco = COCO(ann_file)

    sample_img_urls = []
    sample_img_ids = []
    for i in range(5):
        img_id, img_info = random.choice(list(coco.imgs.items()))
        sample_img_urls.append(img_info['coco_url'])
        sample_img_ids.append(img_id)
    best_captions = capgen.get_captions(sample_img_urls)

    print("-------------------------------------RESULTS-------------------------------------")
    print(f"Getting caption prediction for images: {sample_img_ids}")
    for idx, img_id in enumerate(sample_img_ids):
        references_raw = [caption['caption'] for caption in coco.imgToAnns[img_id]]
        references = [caption['caption'].split() for caption in coco.imgToAnns[img_id]]
        hypothesis = best_captions[idx].split()
        bleu = bleu_score.sentence_bleu(references, hypothesis)
        print(f"ID: {img_id} \n Real caption (1 of 5): {references_raw[0]} \n Sampled caption: {best_captions[idx]} \n BLEU score: {bleu}")


if __name__ == "__main__":
    main()