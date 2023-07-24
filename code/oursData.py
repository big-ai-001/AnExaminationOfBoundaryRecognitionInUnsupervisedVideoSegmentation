# -*- coding: utf-8 -*-
# +
# # %pip install pytube
# # %pip install beautifulsoup4

# +
from pytube import YouTube
from bs4 import BeautifulSoup
# from nltk import windowdiff, pk
import segeval
import numpy as np
import applcation as segapp
import re
import math
import random

pattern = re.compile('\d+:\d+:\d+|\d+:\d+')

# +
# from transformers import RobertaTokenizer, RobertaModel
# import torch

# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# model = RobertaModel.from_pretrained("roberta-base")

# +
# inputs = tokenizer("Hello, my dog is cute",padding=True, truncation=True, return_tensors='pt')#encoded_input
# outputs = model(**inputs, output_hidden_states=True)

# +
# outputs.last_hidden_state

# +
# last_hidden_states

# +
# def max_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
#     return torch.max(token_embeddings, 1)[0]

# +
# def test_max_pooling(model_output, attention_mask):
#     token_embeddings = model_output[-3] #First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
#     return torch.max(token_embeddings, 1)[0]

# +
# # with torch.no_grad():
# model_output = model(**inputs, output_hidden_states=True)

# +
# model_output[-1][-2]

# +
# sentence_embeddings_n1 = max_pooling(model_output, inputs['attention_mask'])
# sentence_embeddings_n2 = test_max_pooling(model_output, inputs['attention_mask'])

# +
# print("Sentence embeddings:")
# print(sentence_embeddings)

# +
# sentence_embeddings_n1

# +
# sentence_embeddings_n2==sentence_embeddings_n1
# -



def sc(t):
    tmp = t.split(':')
    if (len(tmp)==2):
        s = int(tmp[-1])+int(tmp[-2])*60
    else:
        s = int(tmp[-1])+int(tmp[-2])*60+int(tmp[-3])*3600
    return s


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


# -

def dl(url):
    tmp = YouTube(url)
    code = ''
    for i in tmp.captions:
#         print(i.code)
        if 'en' in i.code:
            code = i.code
#     print(code)
    caption = tmp.captions.get_by_language_code(code)
    xml = caption.xml_captions
    with open('./datapool/tmp.srt','w') as f:
        f.write(xml2srt(xml))


# +
# dl('https://www.youtube.com/watch?v=7BHs1BzA4fs')
# -

def xml2srt(text):
    soup = BeautifulSoup(text)                     # 使用 BeautifulSoup 轉換 xml
    ps = soup.findAll('p')                         # 取出所有 p tag 內容

    output = ''                                    # 輸出的內容
    num = 0                                        # 每段字幕編號
    for i, p in enumerate(ps):
        try:
            a = p['a']                             # 如果是自動字幕，濾掉有 a 屬性的 p tag
        except:
            try:
                num = num + 1                      # 每段字幕編號加 1
                text = p.text                      # 取出每段文字
                t = int(p['t'])                    # 開始時間
                d = int(p['d'])                    # 持續時間

                h, tm = divmod(t,(60*60*1000))     # 轉換取得小時、剩下的毫秒數
                m, ts = divmod(tm,(60*1000))       # 轉換取得分鐘、剩下的毫秒數
                s, ms = divmod(ts,1000)            # 轉換取得秒數、毫秒

                t2 = t+d                           # 根據持續時間，計算結束時間
                if t2 > int(ps[i+1]['t']): t2 = int(ps[i+1]['t'])  # 如果時間算出來比下一段長，採用下一段的時間
                h2, tm = divmod(t2,(60*60*1000))   # 轉換取得小時、剩下的毫秒數
                m2, ts = divmod(tm,(60*1000))      # 轉換取得分鐘、剩下的毫秒數
                s2, ms2 = divmod(ts,1000)          # 轉換取得秒數、毫秒


                output = output + str(num) + '\n'  # 產生輸出的檔案，\n 表示換行
                output = output + f'{h:02d}:{m:02d}:{s:02d},{ms:03d} --> {h2:02d}:{m2:02d}:{s2:02d},{ms2:03d}' + '\n'
                output = output + text + '\n'
                output = output + '\n'
            except:
                pass

    return output


# +
def segtag(pack, tList):
    Index = 0
    SL = ''
#     print(pack, tList)
    for i in range(len(pack)):
        if (Index+1 == len(tList)):
            SL+='0'
        elif (sc(pack[i][1][0]) == tList[Index]):
            SL+='1'
            Index+=1
            try:
                print(pack[i][2], '-', pack[i+1][2])
            except:
                pass
        elif (sc(pack[i][1][0]) >= tList[Index]):
            if (abs(sc(pack[i][1][0]) - tList[Index]) > abs(sc(pack[i-1][1][0]) - tList[Index])):
                SL=SL[:-1] + '1'
                SL+='0'
            else:
                SL+='1'
                try:
                    print(pack[i][2], '-', pack[i+1][2])
                except:
                    pass
            Index+=1
        else:
            SL+='0'
            
    SL = '1' + SL[1:]
            
#     print(SL)
    
    slist=[]
    for s in SL.split('1')[1:]:
        slist.append(len(s)+1)
            
    return slist


# +
def rantag(pack, tList):
    SL = '1'*(len(tList)-1)+'0'*(len(pack)-len(tList))
    SL = '1' + ''.join(random.sample(SL,len(SL)))
#     print(SL)
    slist = []
    for s in SL.split('1')[1:]:
        slist.append(len(s)+1)
            
    return slist

def envtag(pack, tList):
    SL = ''
    span = int(len(pack)/len(tList))
    for i in range(len(pack)):
        if (i%span == 0):
            SL+='1'
        else :
            SL+='0'
            
    slist = []
#     print(SL)
    for s in SL.split('1')[1:]:
        slist.append(len(s)+1)
            
    return slist


# -

urlList = ['https://www.youtube.com/watch?v=7BHs1BzA4fs', # freecodepan 计算机和电子
           'https://www.youtube.com/watch?v=XBTJDpT2XaI', # 计算机和电子
           'https://www.youtube.com/watch?v=gfmRrPjnEw4', # 计算机和电子
           'https://www.youtube.com/watch?v=bC7o8P_Ste4', # 计算机和电子
           'https://www.youtube.com/watch?v=8124kv-632k', # 计算机和电子
           'https://www.youtube.com/watch?v=Unzc731iCUY', # MIT OpenCourseWare 工作与教育
           'https://www.youtube.com/watch?v=EH6vE97qIP4', # 互联网和电信
           'https://www.youtube.com/watch?v=tCKk22kaZi4', # 金融
           'https://www.youtube.com/watch?v=L1ung0wil9Y', # 计算机和电子
           'https://www.youtube.com/watch?v=t4K6lney7Zw', # 计算机和电子
           'https://www.youtube.com/watch?v=r6sGWTCMz2k', # 三藍一棕 科学
           'https://www.youtube.com/watch?v=D8NdNt5PNNo', # PBS 美國公視 新闻
           'https://www.youtube.com/watch?v=woXZw4yvW4E', # 新闻
           'https://www.youtube.com/watch?v=qNI4pRDKJFg', # 新闻
           'https://www.youtube.com/watch?v=ZeGZUF_W70Q', # 新闻
           'https://www.youtube.com/watch?v=erLCGSb56KI', # The Daily Show with Trevor Noah 艺术与娱乐
           'https://www.youtube.com/watch?v=P9J0B72bd7U', # 艺术与娱乐
           'https://www.youtube.com/watch?v=cdZZpaB2kDM', # TED 人與社會
           'https://www.youtube.com/watch?v=PQaB0HDdTTg', # Website Learners (AI) 工作與教育
           'https://www.youtube.com/watch?v=FgakZw6K1QQ', # StatQuest (AI) 工作與教育
           'https://www.youtube.com/watch?v=PWcNlRI00jo', # NVIDIA (ceo speech) 計算機和電子
           'https://www.youtube.com/watch?v=JHfI5JbiWhE', # Zach Star(aerospace Engineering) 科學
           'https://www.youtube.com/watch?v=mtvmqI0PV2M', # Economics Explained 人与社会
           'https://www.youtube.com/watch?v=wMGAt4EC77w', # An Honest Discussion About A Universal Basic Income 法律與政府
           'https://www.youtube.com/watch?v=V30VyMMce9s', # No. China Is Not Going To Collapse... Yet 法律與政府
           'https://www.youtube.com/watch?v=5_M2JqeY8CA', # US Military News (military) 新聞
           'https://www.youtube.com/watch?v=UpIpbDRoifw', # US Military News (military) 新聞
           'https://www.youtube.com/watch?v=MkaNleSQapw', # Toby Corban (house) 房地產
           'https://www.youtube.com/watch?v=aj6COIw8vOc', # The Templin Institute (gaming) 遊戲
           'https://www.youtube.com/watch?v=oZJqEkamd4Y', # The Templin Institute (gaming) 遊戲
           'https://www.youtube.com/watch?v=_LEvIpU3Log', # HISTORY(UFO) 科學
           'https://www.youtube.com/watch?v=cjPHSeVMh4M', # Logically Answered (CISCO company) 互聯網和電信
           'https://www.youtube.com/watch?v=2rfRk_mTf7M', # CrashCourse (psychology) 人與社會
           'https://www.youtube.com/watch?v=vxvhGj9fA3g', #Which Healthcare System is Best? Crash Course Public Health #7 健康
           'https://www.youtube.com/watch?v=ngo3ZTrT69A', # Ali the Dazzling(engineering) 興趣愛好與休閒
           'https://www.youtube.com/watch?v=2n66jqRBOd4', # 工作和教育
           'https://www.youtube.com/watch?v=-OmtpjdPhM8', # Enes Yilmazer 房地產
           'https://www.youtube.com/watch?v=-UlxHPIEVqA', # Domain of Science 計算機和電子
           'https://www.youtube.com/watch?v=kFlLzFuslfQ', # Neura Pod – Neuralink (Elon Musk) 興趣愛好和休閒
           'https://www.youtube.com/watch?v=HhomSGnP-x8', # Aleksa Gordić - The AI Epiphany (alphaFold 2) 計算機和電子
          ]

# +
# score = []

# for url in urlList:
#     print(url)
#     yt = YouTube(url)
#     dl(url)
#     pack = load('./datapool/tmp.srt')
#     ref = []
#     print(yt)
#     for i in yt.description.split('\n'):
#         if bool(re.search(pattern, i)):
#             ref.append(sc(re.search(pattern, i).group(0)))
            
#     if (len(ref)==0):
#         continue
        
#     refp = segtag(pack, ref)
    
# #     ranp = envtag(pack, ref)
# #     print(ranp, refp)
# #     score.append([segeval.window_diff(refp, ranp), segeval.pk(refp, ranp), segeval.boundary_similarity(refp, ranp)])

# #     ranp = rantag(pack, ref)
# #     score.append([segeval.window_diff(refp, ranp), segeval.pk(refp, ranp), segeval.boundary_similarity(refp, ranp)])    
    
#     score.append([])
    
#     for i in np.arange(0.25, 1.0, 0.025):
#         mer = []
#         mereslut = segapp.noNSP_roberta("./datapool/tmp.srt", var=i)
# #         print(mereslut)
#         for j in mereslut:
#             if bool(re.search(pattern, j)):
#                 mer.append(sc(re.search(pattern, j).group(0)))
#         try:
#             score[-1].append([segeval.window_diff(segtag(pack, mer), refp), 
#                               segeval.pk(segtag(pack, mer), refp), 
#                               segeval.boundary_similarity(segtag(pack, mer), refp)])            
# #             print(score[-1][-1])
#         except:
#             pass
        
# #         print(math.floor(i), score[-1][-1][0], score[-1][-1][1])

# +
# score = []

# for url in urlList:
#     try:

#     print(url)
#     yt = YouTube(url)
#     dl(url)
#     pack = load('./datapool/tmp.srt')
#     ref = []
#     print(yt)
#     for i in yt.description.split('\n'):
#         if bool(re.search(pattern, i)):
#             ref.append(sc(re.search(pattern, i).group(0)))
            
#     if (len(ref)==0):
#         continue
        
# #     refp = segtag(pack, ref)
    
# #     ranp = envtag(pack, ref)
# #     print(ranp, refp)
# #     score.append([segeval.window_diff(refp, ranp), segeval.pk(refp, ranp), segeval.boundary_similarity(refp, ranp)])

# #     ranp = rantag(pack, ref)
# #     score.append([segeval.window_diff(refp, ranp), segeval.pk(refp, ranp), segeval.boundary_similarity(refp, ranp)])    
    
#     score.append([])
    
#     for i in np.arange(0.25, 1.0, 0.025):
# #     for i in np.arange(0.5, 0.55, 0.025):
#         mer1 = []
#         mer2 = []
#         mereslut1 = segapp.noNSP_roberta("./datapool/tmp.srt", var=i)
#         mereslut2 = segapp.hasNSP_roberta("./datapool/tmp.srt", var=i)
        
#         for j in mereslut1:
#             if bool(re.search(pattern, j)):
#                 mer1.append(sc(re.search(pattern, j).group(0)))
#         for j in mereslut2:
#             if bool(re.search(pattern, j)):
#                 mer2.append(sc(re.search(pattern, j).group(0)))
        
#         tmp1 = segtag(pack, mer1)
#         tmp2 = segtag(pack, mer2)
#         print('mer1', segeval.pk(tmp1, refp))
#         segtag(pack, mer1)
#         print('mer2', segeval.pk(tmp2, refp))
#         segtag(pack, mer2)
        
# #         try:
# #             score[-1].append([segeval.window_diff(segtag(pack, mer), refp), 
# #                               segeval.pk(segtag(pack, mer), refp), 
# #                               segeval.boundary_similarity(segtag(pack, mer), refp)])            
# # #             print(score[-1][-1])
# #         except:
# #             pass
        
# #         print(math.floor(i), score[-1][-1][0], score[-1][-1][1])
# -

score = []
for url in urlList:
    print(url)
    try:
        yt = YouTube(url)
        tmp = yt.description.split('\n')
    except VideoUnavailable:
        continue
    dl(url)
    pack = load('./datapool/tmp.srt')
    ref = []
    for i in tmp:
        if bool(re.search(pattern, i)):
            ref.append(sc(re.search(pattern, i).group(0)))

    if (len(ref)==0):
        continue
    
    print('ref')
    refp = segtag(pack, ref)
        
    for i in np.arange(0.25, 1.0, 0.025):
        mer1 = []
        mer2 = []
        mereslut1 = segapp.noNSP_roberta("./datapool/tmp.srt", var=i)
        mereslut2 = segapp.hasNSP_roberta("./datapool/tmp.srt", var=i)

        for j in mereslut1:
            if bool(re.search(pattern, j)):
                mer1.append(sc(re.search(pattern, j).group(0)))
        for j in mereslut2:
            if bool(re.search(pattern, j)):
                mer2.append(sc(re.search(pattern, j).group(0)))
        
        try:
            print('mer1')
            tmp1 = segtag(pack, mer1)
            print(segeval.pk(tmp1, refp))
            print('mer2')
            tmp2 = segtag(pack, mer2)
            print(segeval.pk(tmp2, refp))
        except:
            continue

score = []
for url in urlList[35:]:
    print(url)
    try:
        yt = YouTube(url)
    except VideoUnavailable:
        continue
    else:
        tmp = yt.description.split('\n')

    dl(url)
    pack = load('./datapool/tmp.srt')
    ref = []
    for i in tmp:
        if bool(re.search(pattern, i)):
            ref.append(sc(re.search(pattern, i).group(0)))

    if (len(ref)==0):
        continue
    
    print('ref')
    refp = segtag(pack, ref)
        
    for i in np.arange(0.25, 1.0, 0.025):
        mer1 = []
        mer2 = []
        mereslut1 = segapp.noNSP_roberta("./datapool/tmp.srt", var=i)
        mereslut2 = segapp.hasNSP_roberta("./datapool/tmp.srt", var=i)

        for j in mereslut1:
            if bool(re.search(pattern, j)):
                mer1.append(sc(re.search(pattern, j).group(0)))
        for j in mereslut2:
            if bool(re.search(pattern, j)):
                mer2.append(sc(re.search(pattern, j).group(0)))
        
        try:
            print('mer1')
            tmp1 = segtag(pack, mer1)
            print(segeval.pk(tmp1, refp))
            print('mer2')
            tmp2 = segtag(pack, mer2)
            print(segeval.pk(tmp2, refp))
        except:
            continue

mer2

# +
dataSet = []
for url in urlList:
    print(url)

#     dl(url)
#     pack = load('./datapool/tmp.srt')
    ref = []
#     while (ref == []):
    print("prosess")
    for i in tmp:
        if bool(re.search(pattern, i)):
            ref.append(sc(re.search(pattern, i).group(0)))
    print("sesses")
    dataSet.append(ref)
# -

len(urlList)

# +
# YouTube("https://www.youtube.com/watch?v=mtvmqI0PV2M").description.split('\n')

# +
# yt.description.split()

# +
# model_name = "bigscience/bloom-2b5"
# model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
# +
# score


# +
# sumWD = 0
# sumPK = 0
# sumB = 0

# for i in score:
#     sumWD += i[0]
#     sumPK += i[1]
#     sumB += i[2]

# print(sumWD/len(score), sumPK/len(score), sumB/len(score))

# # even
# # 0.4855653319144431663680986461 0.4789092746639046694952389167 0.02062008986223592506742241544

# # ramdom
# # 0.5091229070582144287285376464 0.4818107364332174677932294197 0.01242981745196416723132143011

# +
# score[0][0]

# +
# len(urlList)

# +
# np.arange(0.5, 1.0, 0.025)

# +
avgMinWD = 0
avgMinPK = 0
avgMaxB = 0

for i in score:
    MinWD = 1
    MinPK = 1
    MaxB = 0
    
    for j in i:
        if (MinWD > j[0]):
            MinWD = j[0]
        if (MinPK > j[1]):
            MinPK = j[1]
        if (MaxB < j[2]):
            MaxB = j[2]
            
    avgMinWD = avgMinWD + MinWD
    avgMinPK = avgMinPK + MinPK
    avgMaxB = avgMaxB + MaxB
    
avgMinWD = avgMinWD/len(score)
avgMinPK = avgMinPK/len(score)
avgMaxB = avgMaxB/len(score)

# +
avgMinWD
# Decimal('0.3830253509851189459214290786')

# Decimal('0.3783027663185733018381293906')


# +
avgMinPK
# Decimal('0.3545510901704602952092692025')

# Decimal('0.3783027663185733018381293906')


# +
avgMaxB
# Decimal('0.06408585238519482020865215725')

# Decimal('0.09180412111539803549270428769')

# -

score

# +
# windowdiff('00001', '10000', k=5)
# -

