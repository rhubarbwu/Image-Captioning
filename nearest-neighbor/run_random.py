"""
main file to run everything
"""
from caption_generation import CaptionGenerator
from feature_extraction import ImageFeature

from PIL import Image
import matplotlib.pyplot as plt
from nltk.translate import bleu_score
from torchvision.datasets import CocoCaptions
import glob
import random
import warnings
warnings.filterwarnings("ignore")

def main():
    coco_dataset = CocoCaptions(
        root="../data/val2014/val2014", 
        annFile="../annotations/annotations_trainval2014/annotations/captions_val2014.json"
    )

    k = 10
    early_stop = 10000 # set to None to run on entire dataset
    # load_knn = "./knn-models/knn_k=10_num_2000"
    load_knn = None
    capgen = CaptionGenerator(coco_dataset, k=k, early_stop=early_stop, load_knn=load_knn)

    # get images from validation file
    val_dataset = CocoCaptions(
        root="../data/val2014/val2014", 
        annFile="../annotations/annotations_trainval2014/annotations/captions_val2014.json"
    )

    # best_captions = capgen.get_captions(val_dataset)
    sample_imgs = []
    sample_captions = {}
    for i in range(5):
        idx = random.choice(range(len(val_dataset)))
        img_id = val_dataset.ids[idx]
        img, caps = val_dataset[idx]
        sample_imgs.append(img)
        sample_captions[img_id] = caps
    best_captions = capgen.get_captions(sample_imgs)

    print("-------------------------------------RESULTS-------------------------------------")
    print(f"Getting caption prediction for images: {list(sample_captions.keys())}")
    for idx, (img_id, captions) in enumerate(sample_captions.items()):
        references_raw = [caption for caption in captions]
        references = [caption.split() for caption in captions]
        hypothesis = best_captions[idx].split()
        bleu = bleu_score.sentence_bleu(references, hypothesis)
        print(f"ID: {img_id} \n Real caption (1 of 5): {references_raw[0]} \n Sampled caption: {best_captions[idx]} \n BLEU score: {bleu}")


if __name__ == "__main__":
    main()