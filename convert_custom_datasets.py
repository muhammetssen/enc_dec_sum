import pandas as pd

train = pd.read_csv("data/tr_news_raw/train.tsv", sep='\t')
validation = pd.read_csv("data/tr_news_raw/validation.tsv", sep='\t')
test = pd.read_csv("data/tr_news_raw/test.tsv", sep='\t')

'''
train.drop(train.columns.difference(['content', 'abstract']), 1, inplace=True)
validation.drop(validation.columns.difference(['content', 'abstract']), 1, inplace=True)
test.drop(test.columns.difference(['content', 'abstract']), 1, inplace=True)
'''

train.to_csv("data/tr_news_raw/train.csv", index=False)
validation.to_csv("data/tr_news_raw/validation.csv", index=False)
test.to_csv("data/tr_news_raw/test.csv", index=False)