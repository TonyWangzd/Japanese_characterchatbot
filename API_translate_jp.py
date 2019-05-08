
import requests
import hashlib
import random
import json
import pandas as pd
import re
import logging


output_question_file = '/Users/017-010m/code/Japanese_characterchatbot/data/WikiQA-Question.tsv'
output_answer_file = '/Users/017-010m/code/Japanese_characterchatbot/data/WikiQA-Answer.tsv'

jp_question_file = '/Users/017-010m/code/Japanese_characterchatbot/data/JP-Question.tsv'
jp_answer_file = '/Users/017-010m/code/Japanese_characterchatbot/data/JP-Answer.tsv'

en_sentence = 'I am back.'
key = '99sRkrxL4TEdQfU230OO'
m2 = hashlib.md5()

def translate(sentence):
    data = dict()
    data['q'] = sentence
    data['from'] = 'en'
    data['to'] = 'zh'
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
    try:
        if page_json['trans_result']:
            translate_result = page_json['trans_result']
            translate_text = translate_result[0]['dst']
            return translate_text
        else:
            translate_text = '分からないです'
            return translate_text

    except Exception as e:
        print('error',e)



def translate_by_line(file_name, output_file):
    #regex = re.compile(r'/d.+,?')
    file1 = pd.read_csv(file_name, encoding='utf-8', header=None, sep=None)
    for line in file1.iterrows():
        line_text = line[1][1]
        print(line_text)
        jp_line_text = translate(line_text)
        with open(output_file, 'a') as write_file:
            write_file.write(jp_line_text+'\n')
            print(jp_line_text, '\n', line_text)


if __name__ == "__main__":
    try:
        sentence = 'you ass'
        print(translate(sentence))
    except Exception as e:
        print(e)
