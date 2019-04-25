import pandas as pd

output_question_file = '/Users/017-010m/Desktop/data/WikiQA-Question.tsv'
output_answer_file = '/Users/017-010m/Desktop/data/WikiQA-Answer.tsv'

WikiCorpus = '/Users/017-010m/Desktop/data/WikiQA-train.tsv'
file1 = pd.read_csv(WikiCorpus,sep='\t', header=0)
question_set = file1['Question']
answer_set = file1['Sentence']

question_set.to_csv(output_question_file)
answer_set.to_csv(output_answer_file)

print ('done')


