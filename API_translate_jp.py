

import requests
import  re
import urllib.parse
import hashlib
import random
import json

en_sentence = 'I am back.'
key = '99sRkrxL4TEdQfU230OO'
m2 = hashlib.md5()

def translate(sentence):
    data = dict()
    data['q'] = sentence
    data['from'] = 'en'
    data['to'] = 'jp'
    data['appid'] = '20190425000291634'
    data['salt'] = str(random.randint(32768, 65536))

    sign = data['appid'] + data['q'] + data['salt'] + key
    final_sign = hashlib.md5(sign.encode("utf-8")).hexdigest()
    data['sign'] = final_sign
    #print(data['sign'])

    url = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
    page_response = requests.get(url, params=data)
    if page_response.status_code == 200:
        try:
            page_data = page_response.text
            page_json = json.loads(page_data)

        except Exception as e:
            print('error occured', e, 'when getting this page', url)
    else:
        print('cannot get the page')
        return

    if page_json['trans_result']:
        translate_result = page_json['trans_result']
        translate_text = translate_result[0]['dst']
        print(translate_text)

        return translate_text


if __name__ == "__main__":
    translate(en_sentence)