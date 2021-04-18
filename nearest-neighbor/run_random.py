"""
main file to run everything
"""
from caption_generation import CaptionGenerator
from feature_extraction import ImageFeature

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# from nltk.translate import bleu_score
from torchvision.datasets import CocoCaptions
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.bleu.bleu import Bleu
import json
import random
import warnings
warnings.filterwarnings("ignore")

def main():
    ann_file = "../annotations/annotations_trainval2014/annotations/captions_val2014.json" 
    coco_dataset = CocoCaptions(
        root="../data/val2014/val2014", 
        annFile= ann_file
    )

    k = 10
    early_stop = 10000 # set to None to run on entire dataset
    load_knn = "./knn-models/knn_k=10_num_10000"
    res_file = "./results/val2014_results"
    # load_knn = None
    capgen = CaptionGenerator(coco_dataset, k=k, early_stop=early_stop, load_knn=load_knn)

    # get images from validation file
    val_dataset = CocoCaptions(
        root="../data/val2014/val2014", 
        annFile=ann_file
    )
    sample_imgs = []
    sample_img_ids = []
    for i in range(5):
        idx = random.choice(range(len(val_dataset)))
        img_id = val_dataset.ids[idx]
        sample_img_ids.append(img_id)
        img, caps = val_dataset[idx]
        sample_imgs.append(img)

    print(f"Getting caption prediction for images: {sample_img_ids}")
    best_captions = capgen.get_captions(sample_imgs)

    # save results
    results = []
    references = {}
    hypothesis = {}
    for idx, img_id in enumerate(sample_img_ids):
        res_dict = {'image_id': img_id, 'caption':best_captions[idx]}
        ref_caption_ids = capgen.coco.coco.getAnnIds(img_id)
        references[img_id] = {img_id: [capgen.coco.coco.anns[ann_id]['caption'] for ann_id in ref_caption_ids]}
        hypothesis[img_id] = {img_id: [best_captions[idx]]}
        results.append(res_dict)
    with open(res_file, 'w') as f:
        json.dump(results, f)

    print("-------------------------------------RESULTS-------------------------------------")
    # evaluate results
    # coco_result = capgen.coco.coco.loadRes(res_file)
    # cocoEval = COCOEvalCap(capgen.coco.coco, coco_result)
    # cocoEval.params['image_id'] = coco_result.getImgIds()
    # output = cocoEval.evaluate()
    print("--------------------------------------------------------------------------------")
    for idx, img_id in enumerate(sample_img_ids):
        real_caption = references[img_id][img_id][0]
        scores, _ = Bleu().compute_score(references[img_id],hypothesis[img_id])
        bleu = scores[2] # 3-gram
        print(f"ID: {img_id} \n Real caption (1 of 5): {real_caption} \n Sampled caption: {best_captions[idx]} \n BLEU: {bleu}")

if __name__ == "__main__":
    main()