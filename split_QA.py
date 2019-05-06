# coding:utf-8
output_question_file = './data/question.txt'
output_answer_file = './data/answer.txt'

with open('./data/Corpus.txt','r',encoding='utf-8') as file1:
    lines = file1.readlines()
    for line in lines:
        if '- - 'in line:
            with open(output_question_file, 'a',encoding='utf-8') as write_file:
                write_file.write(line.lstrip('- - ')+'\n')
        else:
            with open(output_answer_file, 'a',encoding='utf-8') as write_file:
                write_file.write(line.lstrip('  - ')+'\n')

    print('done')