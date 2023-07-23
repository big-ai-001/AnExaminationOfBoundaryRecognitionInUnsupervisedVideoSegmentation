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

import xml.etree.ElementTree as ET
import pprint
pprint = pprint.pprint


# +
def topic2seg(data):
    fName = data[0]
    Bow = data[1][3:-1]
    Eow = None
    if len(data)==3: Eow=data[2][3:-1]
    
    tree = ET.parse('/workspace/pj/ICSI_plus/Segments/'+fName)
    data = tree.getiterator()
    
#     print(fName)
    
    idList, segList = [], []
    for sIndex in range(1, len(data), 2):
#         print(sIndex)
        key = data[sIndex].attrib['{http://nite.sourceforge.net/}id']
        velue = data[sIndex+1].attrib['href'].split("#")
        velue = [velue[0]] + velue[1].split('..')
        if len(velue)==3:
            velue = [velue[0], velue[1][3:-1], velue[2][3:-1]]
        else:
            velue = [velue[0], velue[1][3:-1]]

        idList.append(key)
        segList.append(velue)
#     print(segList)
    if Eow==None:
#         print(segList[idList.index(Bow)])
        return seg2word(segList[idList.index(Bow)])
    else:
        seg=''
        for i in range(idList.index(Bow), idList.index(Eow)+1):
#             print(segList[i])
            tmp = seg2word(segList[i])
            if tmp != '':
                seg+=tmp+' '
        return seg


# +
def seg2word(data):
    fName = data[0]
    Bow = data[1]
    Eos = None
    if len(data)==3: Eos=data[2]
    
#     tree = ET.parse('/workspace/pj/ICSI_plus/Words/'+"Bdb001.D.words.xml")
    tree = ET.parse('/workspace/pj/ICSI_plus/Words/'+fName)
    data = tree.getiterator()

    idList = []
    wordList =[]

    for i in data[1:]:
    #     print(i.attrib)
        idList.append(i.attrib['{http://nite.sourceforge.net/}id'])
        if (i.tag == 'w'):
            wordList.append(i.text+' ')
        else :
            wordList.append('')
    
#     sentens = ''
    if Eos==None :
        return wordList[idList.index(Bow)].strip()
    else:
        return ''.join(wordList[idList.index(Bow):idList.index(Eos)+1]).strip()


# -

def xmlProssesByPath(path='Bdb001.topic.xml'):
#     ES2002a.topic.xml
    tree = ET.parse('/workspace/pj/ICSI_plus/Contributions/TopicSegmentation/'+path)
    data = tree.getiterator()
    
    sList = []
    pList = []
    for e_index in range(len(data)):
        e = data[e_index]
        if (e.tag == 'topic'):
#             pList += '1'
            if (sList!=[]):
                pList.append(sList)
                sList = []
#             print('topic', e.attrib['description'])
        elif ('child' in e.tag):
            tmp = []
            tmp = e.attrib['href'].split("#")
            tmp = [tmp[0]] + tmp[1].split("..")
            tmp2 = topic2seg(tmp)
            if tmp2!='':
                sList.append(tmp2)
#                 pList += '1'

    if sList!=[]:pList.append(sList)
    msl = ''
    for p in pList:
        msl += '1'
        msl += '0'*(len(p)-1)
        
    return pList, msl


textList, _ = xmlProssesByPath()
textList



