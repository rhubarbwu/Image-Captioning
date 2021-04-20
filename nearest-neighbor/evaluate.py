"""
generate and evaluate NN approach
"""
import tqdm
import json
import pandas as pd

from torchvision.datasets import CocoCaptions
from pycocoevalcap.eval import COCOEvalCap

from caption_generation import CaptionGenerator

train_ann_file = "../annotations/annotations_trainval2014/annotations/captions_val2014.json" 
valid_ann_file = "../annotations/annotations_trainval2014/annotations/captions_val2014.json" 

k = 20
train_early_stop = 20000 # set to None to run on entire dataset
val_early_stop = 1000
load_knn = f"./knn-models/knn_k={k}_num_{train_early_stop}"
# load_knn = None
res_file = f"./results/val2014_k={k}_num_{train_early_stop}_results"
out_file = f"./results/k={k}_trainnum={train_early_stop}_valnum={val_early_stop}_scores.xlsx"

def evaluate():
    coco_dataset = CocoCaptions(
        root="../data/val2014/val2014", 
        annFile= train_ann_file
    )

    capgen = CaptionGenerator(coco_dataset, k=k, early_stop=train_early_stop, load_knn=load_knn)

    # evaluate
    val_dataset = CocoCaptions(
        root="../data/val2014/val2014", 
        annFile=valid_ann_file
    )

    best_captions, cap_map = capgen.evaluate(val_dataset, early_stop=val_early_stop) # best_captions = list(dict('image_id':img_id, 'caption':'caption'))
    with open(res_file, 'w') as f:
        json.dump(best_captions, f)


    # evaluate best captions against gt
    coco_result = capgen.coco.coco.loadRes(res_file)
    cocoEval = COCOEvalCap(capgen.coco.coco, coco_result)
    cocoEval.params['image_id'] = coco_result.getImgIds()
    cocoEval.evaluate()

    indices = ["BLEU 1-gram", "BLEU 2-gram", "BLEU 3-gram", "BLEU 4-gram",
    		   "METEOR", "ROUGE_L", "CIDEr", "SPICE"]
    data = [cocoEval.eval['Bleu_1']] + [cocoEval.eval['Bleu_2']] + [cocoEval.eval['Bleu_3']] + [cocoEval.eval['Bleu_4']] + \
    	   [cocoEval.eval['METEOR']] + [cocoEval.eval['ROUGE_L']] + [cocoEval.eval['CIDEr']] + [cocoEval.eval['SPICE']]
    results = pd.DataFrame(columns=[f"k={k}_Train_num={train_early_stop}_Val_num={val_early_stop}"], index=indices, data=data)
    results.to_excel(out_file)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    evaluate()