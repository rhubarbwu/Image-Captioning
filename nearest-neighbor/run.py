"""
script to run either a random set of samples or user-defined image so we can see the output
"""
from caption_generation import CaptionGenerator
from feature_extraction import ImageFeature

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CocoCaptions
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.bleu.bleu import Bleu
import requests
import sys
import json
import random
import warnings
warnings.filterwarnings("ignore")

def evaluate(sample_img_ids, best_captions, capgen):
    results = []
    references = {}
    hypothesis = {}
    for idx, img_id in enumerate(sample_img_ids):
        res_dict = {'image_id': img_id, 'caption':best_captions[idx]}
        ref_caption_ids = capgen.coco.coco.getAnnIds(img_id)
        references[img_id] = {img_id: [capgen.coco.coco.anns[ann_id]['caption'] for ann_id in ref_caption_ids]}
        hypothesis[img_id] = {img_id: [best_captions[idx]]}
        results.append(res_dict)
    return results, references, hypothesis

def main():
    if len(sys.argv) == 2:
        sample_random = True
        load_knn = sys.argv[1]
    elif len(sys.argv) == 3:
        sample_random = False
        load_knn = sys.argv[1]
        img_path = sys.argv[2]
    else:
        raise Exception(f"Got {len(sys.argv)-1} args, was expecting 1 or 2 (path_to_knn-model, [img_path])")

    ### CHANGE PARAMETERS HERE ###
    train_ann_file = "../annotations/annotations_trainval2014/annotations/captions_train2014.json" 
    valid_ann_file = "../annotations/annotations_trainval2014/annotations/captions_val2014.json" 

    coco_dataset = CocoCaptions(
        root="../data/train2014/train2014", 
        annFile= train_ann_file
    )

    k = int(load_knn.split("knn_k=")[-1].split("_num")[0])
    train_early_stop = int(load_knn.split("_num_")[-1])
    res_file = f"./results/val2014_k={k}_num_{train_early_stop}_results"
    capgen = CaptionGenerator(coco_dataset, k=k, early_stop=train_early_stop, load_knn=load_knn)

    # get images from validation file
    if sample_random :
        val_dataset = CocoCaptions(
            root="../data/val2014/val2014", 
            annFile=valid_ann_file
        )
        sample_imgs = []
        sample_img_ids = []
        for i in range(5):
            idx = random.choice(range(len(val_dataset)))
            img_id = val_dataset.ids[idx]
            sample_img_ids.append(img_id)
            img, caps = val_dataset[idx]
            sample_imgs.append(img)
    else:
        sample_imgs = [img_path]

    img_names = sample_img_ids if sample_random else img_path
    print(f"Getting caption prediction for images: {img_names}")
    best_captions = capgen.get_captions(sample_imgs)

    if sample_random:
        # evaluate and save results
        results, references, hypothesis = evaluate(img_names, best_captions, capgen)

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
            bleu, scores = Bleu(4).compute_score(references[img_id],hypothesis[img_id], verbose=0)
            # bleu = scores[2] # 3-gram
            print(f"ID: {img_id} \n Real caption (1 of 5): {real_caption} \n Sampled caption: {best_captions[idx]} \n BLEU: {bleu}")
    else:
        print(f"Sampled caption: {best_captions[0]}")
        out_path = "results/" + img_path.split("/")[-1]
        if img_path.startswith('http'):
          img = Image.open(requests.get(img_path, stream=True).raw)
        else:
          img = Image.open(img_path)
        plt.axis('off') 
        plt.title(img_path)
        plt.imshow(img)
        plt.figtext(0.5, 0.01, best_captions[0], wrap=True, horizontalalignment='center', fontsize=12)
        plt.savefig(out_path)
        print(f"Output saved to {out_path}")

if __name__ == "__main__":
    main()