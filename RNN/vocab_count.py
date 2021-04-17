from torchvision.datasets.coco import CocoCaptions, CocoDetection
import re
import string


def count_caption_vocab_size(pairs):
    vocab = set()
    for (path2data, path2json) in pairs:
        dataset = CocoDetection(path2data, path2json)
        for (img, target) in dataset:
            for annotation in target:
                sentence = annotation["caption"]
                words = re.sub("[" + string.punctuation + "]", "",
                               sentence).split()
                words_set = set(words)
                vocab = vocab.union(words_set)
    return vocab


train2014 = ("./data/images/train2014",
             "./data/annotations/captions_train2014.json")
train2014_vocab = count_caption_vocab_size([train2014])
print("train2014 has {} unique words.".format(len(train2014_vocab)))  # 31520

val2014 = ("./data/images/val2014", "./data/annotations/captions_val2014.json")
val2014_vocab = count_caption_vocab_size([val2014])
print("val2014 has {} unique words.".format(len(val2014_vocab)))  # 22949

print(len(train2014_vocab.union(val2014_vocab)))  # 37939
