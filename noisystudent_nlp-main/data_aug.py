# Produce data noise at the beginning
import pandas as pd
import numpy as np
import argparse
import json

import pickle

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

# Hyper-parameters
data_set = 'imdb'
noise_type_ls = ['eda','key','back','ocr']
noise_type = noise_type_ls[1]
eda_type = "insert"


if data_set == "sms":
    train_dir = "/home/linzongyu/self-training/ours/data/sms/train_data.txt"
    dev_dir = "/home/linzongyu/self-training/ours/data/sms/dev_data.txt"
    meta_dev_dir = '/home/linzongyu/self-training/ours/data/sms/meta_dev_data.txt'
    test_dir = '/home/linzongyu/self-training/ours/data/sms/test_data.txt'
    unlabel_dir = "/home/linzongyu/self-training/ours/data/sms/unlabeled_data.txt"
elif data_set == 'trec':
    train_dir = "/home/linzongyu/self-training/ours/data/trec/train_data.txt"
    dev_dir = "/home/linzongyu/self-training/ours/data/trec/dev_data.txt"
    meta_dev_dir = '/home/linzongyu/self-training/ours/data/trec/meta_dev_data.txt'
    test_dir = '/home/linzongyu/self-training/ours/data/trec/test_data.txt'
    unlabel_dir = "/home/linzongyu/self-training/ours/data/trec/unlabeled_data.txt"
elif data_set == 'youtube':
    train_dir = "/home/linzongyu/self-training/ours/data/youtube/train_data.txt"
    dev_dir = "/home/linzongyu/self-training/ours/data/youtube/dev_data.txt"
    meta_dev_dir = '/home/linzongyu/self-training/ours/data/youtube/meta_dev_data.txt'
    test_dir = '/home/linzongyu/self-training/ours/data/youtube/test_data.txt'
    unlabel_dir = "/home/linzongyu/self-training/ours/data/youtube/unlabeled_data.txt"
elif data_set == 'youtube_2':
    train_dir = "/home/linzongyu/self-training/ours/data/youtube/train_data_2.txt"
    dev_dir = "/home/linzongyu/self-training/ours/data/youtube/dev_data_2.txt"
    meta_dev_dir = '/home/linzongyu/self-training/ours/data/youtube/meta_dev_data_2.txt'
    test_dir = '/home/linzongyu/self-training/ours/data/youtube/test_data_2.txt'
    unlabel_dir = "/home/linzongyu/self-training/ours/data/youtube/unlabeled_data_2.txt"
elif data_set == 'youtube_5':
    train_dir = "/home/linzongyu/self-training/ours/data/youtube/train_data_5.txt"
    dev_dir = "/home/linzongyu/self-training/ours/data/youtube/dev_data_5.txt"
    meta_dev_dir = '/home/linzongyu/self-training/ours/data/youtube/meta_dev_data_5.txt'
    test_dir = '/home/linzongyu/self-training/ours/data/youtube/test_data_5.txt'
    unlabel_dir = "/home/linzongyu/self-training/ours/data/youtube/unlabeled_data_5.txt"
elif data_set == 'youtube_20':
    train_dir = "/home/linzongyu/self-training/ours/data/youtube/train_data_20.txt"
    dev_dir = "/home/linzongyu/self-training/ours/data/youtube/dev_data_20.txt"
    meta_dev_dir = '/home/linzongyu/self-training/ours/data/youtube/meta_dev_data_20.txt'
    test_dir = '/home/linzongyu/self-training/ours/data/youtube/test_data_20.txt'
    unlabel_dir = "/home/linzongyu/self-training/ours/data/youtube/unlabeled_data_20.txt"
elif data_set == 'ag_news': # _2 is smaller one
    train_dir = "/home/linzongyu/self-training/ours/data/ag_news/train_data_2.txt"
    dev_dir = "/home/linzongyu/self-training/ours/data/ag_news/dev_data_2.txt"
    meta_dev_dir = '/home/linzongyu/self-training/ours/data/ag_news/meta_dev_data_2.txt'
    test_dir = '/home/linzongyu/self-training/ours/data/ag_news/test_data_2.txt'
    unlabel_dir = "/home/linzongyu/self-training/ours/data/ag_news/unlabeled_data_2.txt"
elif data_set == 'imdb':
    train_dir = "/home/linzongyu/self-training/ours/data/imdb50/train_data_shuf.txt"
    dev_dir = "/home/linzongyu/self-training/ours/data/imdb50/dev_data_shuf.txt"
    meta_dev_dir = "/home/linzongyu/self-training/ours/data/imdb50/meta_dev_data_shuf.txt"
    test_dir = "/home/linzongyu/self-training/ours/data/imdb50/test_data_shuf.txt"
    unlabel_dir = "/home/linzongyu/self-training/ours/data/imdb50/unlabeled_data_shuf.txt"
elif data_set == "sms_few":
    train_dir = "/home/linzongyu/self-training/ours/dataset/sms_few/train_data.pkl"
    dev_dir = "/home/linzongyu/self-training/ours/dataset/sms_few/dev_data.pkl"
    meta_dev_dir = '/home/linzongyu/self-training/ours/dataset/sms_few/meta_dev_data.pkl'
    test_dir = '/home/linzongyu/self-training/ours/dataset/sms_few/test_data.pkl'
    unlabel_dir = "/home/linzongyu/self-training/ours/dataset/sms_few/unlabeled_data.pkl"
elif data_set == 'trec_few':
    train_dir = "/home/linzongyu/self-training/ours/dataset/trec_few/train_data.pkl"
    dev_dir = "/home/linzongyu/self-training/ours/dataset/trec_few/dev_data.pkl"
    meta_dev_dir = '/home/linzongyu/self-training/ours/dataset/trec_few/meta_dev_data.pkl'
    test_dir = '/home/linzongyu/self-training/ours/dataset/trec_few/test_data.pkl'
    unlabel_dir = "/home/linzongyu/self-training/ours/dataset/trec_few/unlabeled_data.pkl"
elif data_set == 'ag_news_few': # _2 is smaller one
    train_dir = "/home/linzongyu/self-training/ours/dataset/ag_news_few/train_data.pkl"
    dev_dir = "/home/linzongyu/self-training/ours/dataset/ag_news_few/dev_data.pkl"
    meta_dev_dir = '/home/linzongyu/self-training/ours/dataset/ag_news_few/meta_dev_data.pkl'
    test_dir = '/home/linzongyu/self-training/ours/dataset/ag_news_few/test_data.pkl'
    unlabel_dir = "/home/linzongyu/self-training/ours/dataset/ag_news_few/unlabeled_data.pkl"


dirs = [train_dir,dev_dir,meta_dev_dir,unlabel_dir,test_dir]
datas = [None] * len(dirs)

def data_aug_nlp(text,aug_type):
    if aug_type == "key":
        aug = nac.KeyboardAug()
        augmented_text = aug.augment(text)
    elif aug_type == "eda":
        if eda_type == "insert":
            aug = nac.RandomCharAug(action="insert")
            augmented_text = aug.augment(text)
        elif eda_type == "sub":
            aug = nac.RandomCharAug(action="substitute")
            augmented_text = aug.augment(text)
        elif eda_type == "swap":
            aug = nac.RandomCharAug(action="swap")
            augmented_text = aug.augment(text)
        elif eda_type == "del":
            aug = nac.RandomCharAug(action="delete")
            augmented_text = aug.augment(text)
    elif aug_type == "back":
        back_translation_aug = naw.BackTranslationAug(
            from_model_name='facebook/wmt19-en-de', 
            to_model_name='facebook/wmt19-de-en'
        )
        augmented_text = back_translation_aug.augment(text)
    return augmented_text


def readandaug(manifest):
    write_dir = manifest.replace(".txt","_noisy.csv")
    pkl_dir = manifest.replace(".txt","_noisy.pkl")
    file = open(pkl_dir, "wb")
    data = {}
    data_df = {'index':[],'original_target':[],'src':[],'noisy_src':[],'target':[]}
    with open(manifest,'r',encoding='utf8') as f:
        for line in f.readlines():
            info = json.loads(line.strip())
            # info['src'] = data_aug_nlp(info['src'],noise_type)
            info['noisy_src'] = data_aug_nlp(info['src'],noise_type)
            data[int(info["index"])] = info
            for i,v in info.items():
                data_df[i].append(v)
    pickle.dump(data, file)
    file.close()
    data_df = pd.DataFrame(data_df)
    data_df.to_csv(write_dir,index=False)

def readandaug_pkl(manifest):
    write_dir = manifest.replace(".pkl","_noisy.csv")
    pkl_dir = manifest.replace(".pkl","_noisy.pkl")
    file = open(pkl_dir, "wb")
    data = {}
    data_df = {'index':[],'original_target':[],'src':[],'noisy_src':[],'len':[],'target':[]}
    data_dict = pickle.load(open(manifest,'rb'))
    for i, info in data_dict.items():
        info['noisy_src'] = data_aug_nlp(info['src'],noise_type)
        data[int(info["index"])] = info
        for j,v in info.items():
            data_df[j].append(v)
    pickle.dump(data, file)
    file.close()
    data_df = pd.DataFrame(data_df)
    data_df.to_csv(write_dir,index=False)

for i,tdir in enumerate(dirs):
    if data_set == "sms_few" or data_set == "trec_few" or data_set == "ag_news_few":
        readandaug_pkl(tdir)
    else:
        readandaug(tdir)
