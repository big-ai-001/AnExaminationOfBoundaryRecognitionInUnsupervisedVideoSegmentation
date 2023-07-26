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
# print = pprint.pprint

dt_tree = ET.parse('/workspace/pj/AMI/ontologies/default-topics.xml')
dt_data = dt_tree.getiterator()
defaultTopics = {}

for i in dt_data:
#     print(i.attrib['{http://nite.sourceforge.net/}id'], i.attrib['name'])
    defaultTopics[i.attrib['{http://nite.sourceforge.net/}id']]=i.attrib['name']


def id2seg(listOfInfo):
#     print(listOfInfo)
    fname = listOfInfo[0]
    lenName = len(fname[:-4])
    bos = int(listOfInfo[1][lenName+3:-1])
    eos = None
    if (len(listOfInfo)==3):
        eos = int(listOfInfo[2][lenName+3:-1])+1
#     print(fname, bos, eos)
    
    tree = ET.parse('/workspace/pj/AMI/words/'+fname)
    data = tree.getiterator()
    
    listOfWord = []
    for e in data:
        if (e.tag == 'w'):
            listOfWord.append(e.text)
    return ' '.join(listOfWord[bos:eos])


# +
def xmlProssesByPath(path='ES2002a.topic.xml'):
#     ES2002a.topic.xml
    tree = ET.parse('/workspace/pj/AMI/topics/'+path)
    data = tree.getiterator()
    
    sList = []
    pList = []
    for e_index in range(len(data)):
        e = data[e_index]
        if (e.tag == 'topic'):
            if sList!=[]:pList.append(sList)
            sList = []
#             if (e.get('other_description') == None):
#                 print(defaultTopics[data[e_index+1].attrib['href'][22:-1]])
#             else :
#                 print(e.get('other_description'))
        elif ('child' in e.tag) :
            tmp = e.attrib['href'].split("#")
            tmp = [tmp[0]] + tmp[1].split("..")
            tmp = id2seg(tmp)
            if tmp!= '':sList.append(tmp)
    #     print(e.tag)
    
    if sList!=[]:pList.append(sList)
    
#     SL = ''
    SL = []
    for s_l in pList:
        SL.append(len(s_l))
#         SL+='1'
#         SL+='0'*(len(s_l)-1)
#     print(SL)
    
    return [pList, SL]
# -


