from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import SRTFormatter
from pytube import YouTube

def dl(url):
    formatter = SRTFormatter()
    transcript = YouTubeTranscriptApi.get_transcript(YouTube(url).video_id)
    transcript = formatter.format_transcript(transcript)
    with open('./datapool/tmp.srt', 'w', encoding='utf-8') as f:
        f.write(transcript)

# # -*- coding: utf-8 -*-
# # +
# # # %pip install pytube
# # # %pip install beautifulsoup4
# # -

# from pytube import YouTube
# from bs4 import BeautifulSoup
# # import re

# # +
# # pattern = re.compile('\d+:\d\d:\d\d|\d\d:\d\d')

# # +
# # yt = YouTube('https://www.youtube.com/watch?v=7BHs1BzA4fs')

# # +
# # for i in yt.description.split('\n'):
# #     if bool(re.search(pattern, i)):
# #         print(re.search(pattern, i).group(0), re.split(' ', re.split(pattern, i, 1)[-1], 1)[-1])

# # +
# # print(bool(re.search('\d\d:\d\d', 'AQEWRSVQAWETB 07:09')))

# # +
# # for i in yt.streams.filter(progressive=True):
# #     print(i)

# # +
# # stream = yt.streams.get_by_itag(22)

# # +
# # stream.download(filename='tmp.mp4')
# # -

# def dl(url):
#     tmp = YouTube(url)
#     code = ''
#     for i in tmp.captions:
# #         print(i.code)
#         if 'en' in i.code:
#             code = i.code
# #     print(code)
#     assert code != '', f'無字幕: {tmp.captions}'
#     caption = tmp.captions.get_by_language_code(code)
#     assert caption != None, '無字幕'
#     xml = caption.xml_captions
#     with open('./datapool/tmp.srt','w') as f:
#         f.write(xml2srt(xml))


# def xml2srt(text):
#     soup = BeautifulSoup(text)                     # 使用 BeautifulSoup 轉換 xml
#     ps = soup.findAll('p')                         # 取出所有 p tag 內容

#     output = ''                                    # 輸出的內容
#     num = 0                                        # 每段字幕編號
#     for i, p in enumerate(ps):
#         try:
#             a = p['a']                             # 如果是自動字幕，濾掉有 a 屬性的 p tag
#         except:
#             try:
#                 num = num + 1                      # 每段字幕編號加 1
#                 text = p.text                      # 取出每段文字
#                 t = int(p['t'])                    # 開始時間
#                 d = int(p['d'])                    # 持續時間

#                 h, tm = divmod(t,(60*60*1000))     # 轉換取得小時、剩下的毫秒數
#                 m, ts = divmod(tm,(60*1000))       # 轉換取得分鐘、剩下的毫秒數
#                 s, ms = divmod(ts,1000)            # 轉換取得秒數、毫秒

#                 t2 = t+d                           # 根據持續時間，計算結束時間
#                 if t2 > int(ps[i+1]['t']): t2 = int(ps[i+1]['t'])  # 如果時間算出來比下一段長，採用下一段的時間
#                 h2, tm = divmod(t2,(60*60*1000))   # 轉換取得小時、剩下的毫秒數
#                 m2, ts = divmod(tm,(60*1000))      # 轉換取得分鐘、剩下的毫秒數
#                 s2, ms2 = divmod(ts,1000)          # 轉換取得秒數、毫秒

#                 output = output + str(num) + '\n'  # 產生輸出的檔案，\n 表示換行
#                 output = output + f'{h:02d}:{m:02d}:{s:02d},{ms:03d} --> {h2:02d}:{m2:02d}:{s2:02d},{ms2:03d}' + '\n'
#                 output = output + text + '\n'
#                 output = output + '\n'
#             except:
#                 pass

#     return output


