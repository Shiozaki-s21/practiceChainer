import chainer
import chainer.functions as func #F
import chainer.links as links #L
from   chainer import training, datasets, iterators, optimizers
from   chainer.training import extensions
import numpy as np

import codecs
import re
import urllib.parse
import urllib.request
import os
import socket
from PIL import Image

# 保存場所の作成
if not os.path.isdir('portrait'):
    os.mkdir('portrait')
if not os.path.isabs('train'):
    os.mkdir('train')

# URLのリスト
base_url = 'https://commons.wikimedia.org'
url = base_url + '/wiki/Category:17th-century_oil_portraits_of_standing_women_at_three-quarter_length'
suburl = base_url + '/wiki/File:'
next_page = url

# タイムアウトを設定
socket.setdefaulttimeout(10)

# 画像サイズの上限を廃止
Image.MAX_IMAGE_PIXELS = None

# スクレイピング
while len(next_page) > 0:
    url = next_page
    next_page = ''

    #日本語のWikipediaのページ
    with urllib.request.urlopen(url) as response:
        html = response.read().decode('utf-8')

        title - re.findall(r'<title>([¥s¥S]*)) - Wikimedeia Commons</title>', html)

        if lem(title) < 1 :
            break

        nextpage = re.findall(¥r'<a¥s*href=¥"(/w/index.php?[¥s¥S]*)¥" title=¥"' + title[0] +'¥">[¥s¥S]*>next page</a>',¥html)


        #re.findall(¥r'<div class = ¥"gallerytext¥">¥s + <a¥s + href = ¥"/wiki/File:(¥S*)¥"',¥html, re .DOTALL>)
