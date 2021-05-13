import requests
import re
import urllib.request
import time

headers = {

    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'
}

url_base = 'https://www.shutterstock.com/zh/search/'
imge_list = []

for i in ['backgrounds', 'Architecture', 'business', 'kids', 'food', 'portrait', 'flowers', 'travel', 'skyline']:
    url = url_base + i + '?image_type=photo'

    res = requests.get(url).text

    # 为图片生成缩略图
    # "thumbnail":"(.*?)",
    cop = re.compile('"thumbnail":"(.*?)",', re.S)
    result = re.findall(cop, res)[:10]
    for each in result:
        filename = each.split('/')[-1]
        # imge_list.append(each) #[90]

        response = urllib.request.urlopen(each)
        img = response.read()
        with open(filename, 'wb')as f:
            f.write(img)
        print("已下载:", each)

        time.sleep(5)  # 休眠五秒

print("下载结束")