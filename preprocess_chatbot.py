# coding:utf-8
output_question_file = './data/question.txt'
output_answer_file = './data/answer.txt'
file = './data/chatterbot.tsv'


with open(file, 'r', encoding='utf-8') as file1:
    lines = file1.readlines()
    for line in lines:
        corpus = line.split('\t')
        if len(corpus[0])>0:
            with open(output_question_file, 'a',encoding='utf-8') as write_file:
                write_file.write(corpus[0].rstrip('。')+'\n')
        if len(corpus[1])>0:
            with open(output_answer_file, 'a',encoding='utf-8') as write_file:
                write_file.write(corpus[1].rstrip('。'))


