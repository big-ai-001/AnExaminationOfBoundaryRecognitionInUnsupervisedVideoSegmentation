# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# git clone https://github.com/abhirooptalasila/AutoSub
# $ cd AutoSub

# +
# import torch
# torch.cuda.is_available()

# +
# # %pip install pytextrank
# # %pip install bertopic

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# +
# from bertopic import BERTopic
# from sklearn.datasets import fetch_20newsgroups
 
# docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

# topic_model = BERTopic()
# topics, probs = topic_model.fit_transform(docs)

# +
# import spacy
# nlp = spacy.load('en_core_web_trf')
# spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

# +
# with open('./stopwords.txt', "w") as f:
#     for i in spacy_stopwords:
#         f.write(i+"\n")
    
# # with open("output.txt", "w") as f:
# #   f.write("這是要寫入的資料\n")
# #   f.write("這是第二行資料\n")

# +
# import os
import numpy as np
import pandas as pd
# import nltk
import re
# import math
import copy
import spacy
# import pytextrank
from summarizer import Summarizer
# from tensorflow.keras import backend
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity as cosine
# from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import uuid

from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array
# from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import BartTokenizer, BartForConditionalGeneration
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# os.environ["TF_KERAS"]="1"
# smtokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
# smmodel = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

# +
#RRoberta
from transformers import RobertaTokenizer, RobertaModel
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

R_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
R_model = RobertaModel.from_pretrained("roberta-base").to(device)

# +
config_path = "./model/bert_config.json"
checkpoint_path = './model/bert_model.ckpt'
dict_path = './model/vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)

model_nsp = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_nsp=True)
model_sim = build_transformer_model(config_path, checkpoint_path)

bartmodel = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
barttokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# bartmodel = BartForConditionalGeneration.from_pretrained("facebook/bart-large-xsum")
# barttokenizer = BartTokenizer.from_pretrained("facebook/bart-large-xsum")

# KeyBART = AutoModelForSeq2SeqLM.from_pretrained("bloomberg/KeyBART")
# KeyBartTokenizer = AutoTokenizer.from_pretrained("bloomberg/KeyBART")

nlp = spacy.load('en_core_web_trf')
lemmatizer = WordNetLemmatizer()
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
# nlp.add_pipe("textrank")

sbert = Summarizer()


# -

def cosine_similarity(v1, v2): # cos夾角
    return dot(v1, v2)/(norm(v1)*norm(v2))


# +
def load(path):
    data = []
    with open(path, 'r', encoding='UTF-8') as i:
        data=i.readlines()
#     data = pd.read_csv(path, sep='\n')
#     data = np.array(data)
    
    tmp =[]
#     tmp.append(np.array(['0']))
    
    for line in data:
        tmp.append(line.strip())
    
#     print(tmp)
    
    subPack = [] #[[Number], [Time], [字幕]]
    pack = [] #[Pack1, Pack2, Pack3 ...]
    for line in tmp:
        if (len(subPack)<3):
            subPack.append(line)
        else:
            subPack[-1]=subPack[-1] + ' ' + line
        if(line == ''):
            if (len(subPack)>=3):
                pack.append(subPack)
            subPack = []
#     print(pack)
            
    # replace [00:00:00,000 --> 00:00:06] to [[00:00:00], [00:00:06]]
    for i in range(len(pack)):
        s = ""
#         print(pack[i])
        s = pack[i][1]
        s = s.replace(" --> ", "")
        s = s[0:8], s[12:20]
        a = list(s)
        pack[i][1] = a
        
    return pack


# +
def max_pooling(model_output, attention_mask):
    token_embeddings = model_output[-1][-2] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    return torch.max(token_embeddings, 1)[0]

def roberta_seq2par(packList, var=0.6): # packList = [[text, time, keyword], ...]
#     var=0.6# 正比
#     var=0.7# 正比
    
    # 計算 seq2vec
    seq2vecList = []
    for i in range(len(packList)):
        # print(packList[i][0])
        inputs = R_tokenizer(packList[i][0],padding=True, truncation=True, return_tensors='pt').to(device)
        model_output = R_model(**inputs, output_hidden_states=True)
        sentence_embeddings = max_pooling(model_output, inputs['attention_mask'])
        seq2vecList.append(sentence_embeddings[0].cpu().detach().numpy())
        
    # 計算 cosine_similarity
    cosimList = []
    for i in range(len(seq2vecList)-1):
        cosimList.append(cosine_similarity(seq2vecList[i], seq2vecList[i+1]))
        
    # 計算合併時間列表
    idxList = []
    lessThanVar = [] 
    lessThanVar.append('00:00:00')
    for i in range(len(cosimList)):
        if(cosimList[i] < var):
            idxList.append(i)
            lessThanVar.append(packList[i+1][1])  # 段落起始時間
            
    combineText = []
    tmpText = ""
    for i in range(len(cosimList)):
        if len(tmpText) == 0: # 新段落開始
            tmpText = packList[i][0]
        
        if (cosimList[i] > var): # 高相似段落合併
            tmpText += ' '+packList[i+1][0]
        else:  # 低相似段落分割
            combineText.append(tmpText)
            tmpText = ""
    
        if (i == len(packList)-2):  # 處理短尾問題
            if len(tmpText) != 0:
                combineText.append(tmpText)
            else:
                combineText.append(packList[i+1][0])
    
    return list(zip(combineText, lessThanVar))


# -

# Use NSPsim 2do line2seq
def line2seq(pack, var = 0.6):
#     var = 0.6 # NSPsim閥值，和段落數成正比
#     var = 0.7 # NSPsim閥值，和段落數成正比
    
    # Text2Token
    token_ids = ""
    segment_ids = ""
    result = [] # [NSPsim1, NSPsim2, NSPsim ...]
    for i in range(len(pack)-1):
        token_ids, segment_ids = tokenizer.encode(
            pack[i][2], pack[i+1][2], maxlen = 512, truncate_from='left')
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        #計算nsp相似性
        probas = model_nsp.predict([token_ids, segment_ids])[0]
        result.append(probas[0])
#     print(result)
        
    # add2pack pack = [[Number], [Time], [字幕], [NSPsim]]
    for i in range(len(pack)-1):
        pack[i].append(result[i])
#     print(pack)
    
    idxList = []
    lessThanVar = [] 
    lessThanVar.append('00:00:00')

    for i in range(len(pack)-1):
        if(pack[i][3] < var):
            idxList.append(i)
            lessThanVar.append(pack[i][1][1])  # 段落起始時間 = 分割段落結束時間
#     print(lessThanVar)
    
    combineText = []
    tmpText = ""
    for i in range(len(pack)-1):
        if len(tmpText) == 0: # 新段落開始
            tmpText = pack[i][2]
        
        if (pack[i][3] >= var): # 高相似段落合併
            tmpText += ' '+pack[i+1][2]
        else:  # 低相似段落分割
            combineText.append(tmpText)
            tmpText = ""
    
        if (i == len(pack)-2):  # 處理短尾問題
            if len(tmpText) != 0:
                combineText.append(tmpText)
            else:
                combineText.append(pack[i+1][2])
                
    return list(zip(combineText, lessThanVar))
#   combineText_copy = copy.deepcopy(combineText)


def seq2par(packList, var=0.6): # packList = [[text, time, keyword], ...]
#     var=0.6# 正比
#     var=0.7# 正比
    
    # 計算 seq2vec
    seq2vecList = []
    for i in range(len(packList)):
        # print(packList[i][0])
        token_ids, segment_ids = tokenizer.encode(packList[i][0], maxlen = 512, truncate_from='right')
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        seq2vecList.append(model_sim.predict([token_ids, segment_ids])[0][0])
        
    # 計算 cosine_similarity
    cosimList = []
    for i in range(len(seq2vecList)-1):
        cosimList.append(cosine_similarity(seq2vecList[i], seq2vecList[i+1]))
        
    # 計算合併時間列表
    idxList = []
    lessThanVar = [] 
    lessThanVar.append('00:00:00')
    for i in range(len(cosimList)):
        if(cosimList[i] < var):
            idxList.append(i)
            lessThanVar.append(packList[i+1][1])  # 段落起始時間
            
    combineText = []
    tmpText = ""
    for i in range(len(cosimList)):
        if len(tmpText) == 0: # 新段落開始
            tmpText = packList[i][0]
        
        if (cosimList[i] > var): # 高相似段落合併
            tmpText += ' '+packList[i+1][0]
        else:  # 低相似段落分割
            combineText.append(tmpText)
            tmpText = ""
    
        if (i == len(packList)-2):  # 處理短尾問題
            if len(tmpText) != 0:
                combineText.append(tmpText)
            else:
                combineText.append(packList[i+1][0])
    
    return list(zip(combineText, lessThanVar))


# +
def tfidf(pack, uuid):
#     print(pack)
    combineText = []
    for i in pack:
        combineText.append(i[0])
#     combineText = zip(*pack)[0]
    
    #預處理
    for i in range(len(combineText)): 
        combineText[i] = combineText[i].lower()
        combineText[i] = re.sub(r"[^\w\s]", "", combineText[i])
        combineText[i] = re.sub(r"[0-9]", "", combineText[i])
    
    stopWords = set(stopwords.words("english"))
    stopWords = stopWords.union(spacy_stopwords).union({'right', 'going', 'um', 'uh'})
    passTag = {"FW", "NN", "NNS", "NNP", "NNPS"}
    words_filtered = [] # 分詞後的全文本
    for word in combineText:
        # 段詞
        words = word_tokenize(word)
        pos_tagged_sent = pos_tag(words)
        
        for wIndex in range(len(pos_tagged_sent)):
            w = pos_tagged_sent[wIndex][0]
            tag = pos_tagged_sent[wIndex][1]
            if (w not in stopWords)and(tag in passTag): # 去停用詞
                words_filtered.append(w)
    # 計算TF
    count = {}
    for words in words_filtered:
        if words in count:
            count[words] += 1
        else:
            count[words] = 1
            
    # tf =出現在所有文檔的某個詞 / 全部文檔的詞數(包含重複)
    # idf = log(所有文檔 / 某詞出現在某a個文檔) (以10為底)
    # tf * idf = tf-idf
    
    keys = list(count.keys())
    tf_freq = []
    for i in range(len(count)):
        tf_freq.append(count[keys[i]]/len(words_filtered))
     
    # 計算IDF
    idf_list = {}
    for key in keys:
        idf_list[key] = 0
        for word in combineText:
            if key in word:
                idf_list[key] += 1
    
    inverse_document_freqency = []
    for i in range(len(idf_list)):
        inverse_document_freqency.append(len(combineText)/idf_list[keys[i]])
      
    # TF * IDF
    tfidf = np.array(tf_freq)*np.array(inverse_document_freqency)
    tfidf_dict = {}
    for i in range(len(keys)):
        tfidf_dict[keys[i]] = tfidf[i]
       
    # WordCloud
    tmpCloud = WordCloud(width=1200,height=400).fit_words(tfidf_dict) 
    tmpCloud.to_file("./wc/"+uuid+".png")
    
        
    # sorted
    tfidfResult = {k: v for k, v in sorted(
        tfidf_dict.items(), key=lambda item: item[1], reverse=True)}
    
    # TopN
    TopNTfResult = list(tfidfResult.keys())[:100]
    
#     print(TopNTfResult)
        
    # wst = [[pack1], [pack2], [pack3] ,...] [packn] = [word1, word2, word3 ,...]
    wst = [] 
    for sentence in combineText:
        seq_list = word_tokenize(sentence)
        wst.append(seq_list)
    
    keyWord = []
    time_lest = 0
    for i in range(len(wst)):
#         print(pack[i][1])
        ws = wst[i]
        spl = pack[i][1].split(":")
        time_now = int(spl[0])*60*60+int(spl[1])*60+int(spl[2])
        if (time_now-time_lest<=30 and time_now!=0): # 太短砍掉
            continue
        spl = str(int(spl[0])*60+int(spl[1]))+":"+spl[2]
        tmp = []
        for keyw in TopNTfResult:
            if keyw in ws:
                tmp.append(keyw)
            if len(tmp)==5:
                break
        time_lest = time_now
        if len(tmp)!=0:
            keyWord.append(spl.zfill(5)+' '+', '.join(tmp))
    
#     # TF = [[0, 0 , 0 ,... ,0],[0, 0 , 0 ,... ,0],... ] shape(packN, 20)
#     TF = np.zeros((np.shape(wst)[0], 20)) 
#     for i in range(len(Top20TfResult)):  # Top20TfResult 循環
#         for j in range(len(wst)):  # wst 循環
#             for k in range(len(wst[j])):  # 找出 wst[j] 中出現了那些 top20keyWord
#                 if(Top20TfResult[i] == wst[j][k]):
#                     TF[j][i] = 1
#                     continue
                    
#     # 列出keyWord分布
#     keyWord = []
#     tmp = []
#     for i in TF:
#         for j in range(len(i)):
#             if(i[j]):
#                 tmp.append(Top20TfResult[j])
#         keyWord.append(tmp)
#         tmp = []
    
    return keyWord


# +
# def sm(packList):
#     sm_data = []
#     for i in packList:
#         spl = i[1].split(":")
#         spl = str(int(spl[0])*60+int(spl[1]))+":"+spl[2]
        
#         summary = ''.join(smtokenizer.decode(smmodel.generate(**smtokenizer(i[0], truncation=True, padding="longest", return_tensors="pt"))[0]))
#         sm_data.append(spl.zfill(5)+' '+summary)
        
#     return sm_data
# -

def sbertm(packList, lenvar=50):
    time_lest = 0
    time_future = 0
    sm_data = []
    for Index in range(len(packList)):
        i = packList[Index];
        print(i)
        spl = i[1].split(":")
        time_now = int(spl[0])*60*60+int(spl[1])*60+int(spl[2])
        spl = str(int(spl[0])*60+int(spl[1]))+":"+spl[2]
        
        if (len(packList) == Index+1):
            time_future = time_now+31 #強制pass
        else:    
            splF = packList[Index+1][1].split(":")
            time_future = int(splF[0])*60*60+int(splF[1])*60+int(splF[2])
        
        if (time_future-time_now<=30 and time_now!=0): # 太短砍掉
            continue
        
        summary = ''.join(sbert(i[0]))
        #print("Pre summary ",summary)
        if (len(summary) == 0):
            summary = i[0]
        if (len(summary) >= lenvar):
            inputs = barttokenizer(summary, max_length=1024, return_tensors="pt")
            summary_ids = bartmodel.generate(inputs["input_ids"], num_beams=3, max_length=lenvar)
            summary = barttokenizer.batch_decode(summary_ids, skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        #clean_up_tokenization_spaces=False
        if (len(summary)<=10): # 太短砍掉
            continue
        time_lest = time_now
        sm_data.append(spl.zfill(5)+' '+summary)
        #print("Pos summary ",summary)
    
    return sm_data


# 處理並丟出
def onlyTextProsess(pack, var=0.6):
    token_ids = ""
    segment_ids = ""
    result = [] # [NSPsim1, NSPsim2, NSPsim ...]
    for i in range(len(pack)-1):
        token_ids, segment_ids = tokenizer.encode(
            pack[i], pack[i+1], maxlen = 512, truncate_from='left')
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        #計算nsp相似性
        probas = model_nsp.predict([token_ids, segment_ids])[0]
        result.append(probas[0])
#     for i in range(len(pack)-1):
#         pack[i].append(result[i])
        
    combineText = []
    tmpText = ""
    for i in range(len(result)):
        if len(tmpText) == 0: # 新段落開始
            tmpText = pack[i]
        if (result[i] >= var): # 高相似段落合併
            tmpText += ' '+pack[i+1]
        else:  # 低相似段落分割
            combineText.append(tmpText)
            tmpText = ""
    
        if (i == len(pack)-2):  # 處理短尾問題
            if len(tmpText) != 0:
                combineText.append(tmpText)
            else:
                combineText.append(pack[i+1])
    
    packList = combineText
    
    seq2vecList = []
    for i in range(len(packList)):
        token_ids, segment_ids = tokenizer.encode(packList[i], maxlen = 512, truncate_from='right')
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        seq2vecList.append(model_sim.predict([token_ids, segment_ids])[0][0])

    cosimList = []
    for i in range(len(seq2vecList)-1):
        cosimList.append(cosine_similarity(seq2vecList[i], seq2vecList[i+1]))
            
    combineText = []
    tmpText = ""
    for i in range(len(cosimList)):
        if len(tmpText) == 0: # 新段落開始
            tmpText = packList[i]
        
        if (cosimList[i] > var): # 高相似段落合併
            tmpText += ' '+packList[i+1]
        else:  # 低相似段落分割
            combineText.append(tmpText)
            tmpText = ""
    
        if (i == len(packList)-2):  # 處理短尾問題
            if len(tmpText) != 0:
                combineText.append(tmpText)
            else:
                combineText.append(packList[i+1])
    
    return combineText


# +
# def kbertm(packList, lenvar=50):
#     sm_data = []
#     for i in packList:
#         spl = i[1].split(":")
#         spl = str(int(spl[0])*60+int(spl[1]))+":"+spl[2]
#         summary = ''.join(sbert(i[0]))
#         inputs = KeyBartTokenizer(summary, max_length=1024, return_tensors="pt")
#         summary_ids = KeyBART.generate(inputs["input_ids"], num_beams=3, max_length=lenvar)
#         summary = KeyBartTokenizer.batch_decode(summary_ids, skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
#         sm_data.append(spl.zfill(5)+' '+summary)
#         #print("Pos summary ",summary)
#     return sm_data

# +
# def topic(packList):
#     docs=[]
#     for i in packList:
#         if len(i[0])!=0:
#             docs.append(i[0])
# #     print(docs)
# #     docs = [i[0] if len(i[0])!=0]
# #     topic_model = BERTopic()
#     topics, probs = topic_model.transform(docs)
#     sm_data = []
#     for i in range(len(packList)):
#         if topics[i]==-1:
#             continue
#         spl = packList[i][1].split(":")
#         spl = str(int(spl[0])*60+int(spl[1]))+":"+spl[2]
# #         print(i[0])
#         summary = ' '.join([i[0] for i in topic_model.get_topic(topics[i])])
#         sm_data.append(spl.zfill(5)+' '+summary)
#     return sm_data
# -

# 處理並丟出
def prosess(path, var=0.75, lenvar=50):
    data = load(path)
    data = line2seq(data, var=var)
    data = seq2par(data, var=var)
    my_uuid = str(uuid.uuid4())
    tfidf_data = tfidf(data, my_uuid)
#     sm_data = sm(data)
#     print(data)
    sbert_data = sbertm(data, lenvar=lenvar)
#     print(sbert_data)
#     topic_data = topic(data)
    return [tfidf_data, sbert_data, my_uuid]


def noNSP_roberta(path, var=0.75
#                   , lenvar=50
                 ):
#     data = load(path)
#     data = line2seq(data, var=1.0)
#     data = roberta_seq2par(data, var=var)
# #     print(data)
#     tfidf_data = tfidf(data)
# #     sm_data = sm(data)
# #     sbert_data = sbertm(data, lenvar=lenvar)
# #     print(sbert_data)
# #     topic_data = topic(data)
    data = load(path)
    data = line2seq(data, var=1)
    data = seq2par(data, var=var)
    my_uuid = str(uuid.uuid4())
    tfidf_data = tfidf(data, my_uuid)
    return tfidf_data


def hasNSP_roberta(path, var=0.75
#                   , lenvar=50
                 ):
#     data = load(path)
#     data = line2seq(data, var=1.0)
#     data = roberta_seq2par(data, var=var)
# #     print(data)
#     tfidf_data = tfidf(data)
# #     sm_data = sm(data)
# #     sbert_data = sbertm(data, lenvar=lenvar)
# #     print(sbert_data)
# #     topic_data = topic(data)
    data = load(path)
    data = line2seq(data, var=var)
    data = seq2par(data, var=var)
    my_uuid = str(uuid.uuid4())
    tfidf_data = tfidf(data, my_uuid)
    return tfidf_data


def test(path, var=0.75, lenvar=50):
    data = load(path)
    data = line2seq(data, var=var)
    data = seq2par(data, var=var)
#     print(data)
    tfidf_data = tfidf(data)
#     sm_data = sm(data)
#     sbert_data = sbertm(data, lenvar=lenvar)
#     print(sbert_data)
#     topic_data = topic(data)
    return tfidf_data


def noNSP(path, var=0.75, lenvar=50):
    data = load(path)
    data = line2seq(data, var=1.0)
    data = seq2par(data, var=var)
#     print(data)
    tfidf_data = tfidf(data)
#     sm_data = sm(data)
#     sbert_data = sbertm(data, lenvar=lenvar)
#     print(sbert_data)
#     topic_data = topic(data)
    return tfidf_data

# +
# path = u"./dataset/s2t/EN_0304_1.srt"
# # # path = u"./dataset/s2t/EN_0304_1.srt"
# data = load(path)
# data = line2seq(data)
# data = seq2par(data)
# print(sbertm(data))

# +
# tfidf(data)

# +
# from bertopic import BERTopic
# docs = [i[0] for i in data]
# topic_model = BERTopic()
# topics, probs = topic_model.fit_transform(docs)

# +
# # docs
# from sklearn.datasets import fetch_20newsgroups
# docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
# len(docs)
# -

#for i in prosess(path):
     #print(i)

# +
# len('''You can use squams, the scrams more formal. You have lots of program and, um, you have a fix schedule by the bend on chart, because you're going to finish that in a very, very short time.
# 07:35 We regarded the shot, brian or another thing.''')

# +
#print(summarizer('''What you what you think about if I said that's on is 20 min is short? Is that is a small company, one year, probably soon. Not the same, but it is something in common. And in part of that is a different expression, right? It focused on process, a unified process. We have the right what we need to do that. So by using that,  because we have a very complex system. So you're not going to use up  right to to design a very tiny project. Something about IP And they help you understand that IBM is an international company. One story that many branch by chance to wear a lot of store again. So group of professional comes from different areas need to talk about, but out of thing, because it's a very big All right, I need my test some function. All right, and to this, so that, and then go forward. So extend a modified version, extension that because you're going to see here production, what's Let me see which one. Keep operation, the operation, like the contents. So that is the extension of unified process, where it combines to two books together. We know the accounting is quite complex, even this small company.''', max_length=100, min_length=30, do_sample=False))
# +
# prosess(u"./dataset/CNNResult/001.srt")

# +
# prosess(u"./dataset/s2t/EN_0304_1.srt")


# +
# len("Suppose therefore we have given a")

# +
# print('00:00 Unsupervised feature based approaches  Learning widely applicable report  Representations of words has been  an active area of research for decades. Unsupervised\xa0features\xa0based approaches are a new way to teach people how to use words. Learning widely\xa0 applicable report is based on a study of more than 1,000 words.'.strip())

# +
# import spacy
# import pytextrank

# # example text
# text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types."

# # load a spaCy model, depending on language, scale, etc.
# nlp = spacy.load("en_core_web_sm")

# # add PyTextRank to the spaCy pipeline
# nlp.add_pipe("textrank")
# doc = nlp(text)

# # examine the top-ranked phrases in the document
# for phrase in doc._.phrases:
#     print(phrase.text)
#     print(phrase.rank, phrase.count)
#     print(phrase.chunks)
# -


