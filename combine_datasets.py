import os

import pandas as pd
from datasets import load_dataset

# TR-News dataset variables
data_files = dict()
data_files["train"] = "data/tr_news_raw/train.csv"
data_files["validation"] = "data/tr_news_raw/validation.csv"
data_files["test"] = "data/tr_news_raw/test.csv"

tr_news_dataset = load_dataset("csv", data_files=data_files)
ml_sum_dataset = load_dataset("mlsum", "tu")

tr_news_train_size = str(len(tr_news_dataset['train']))
tr_news_val_size = str(len(tr_news_dataset['validation']))
tr_news_test_size = str(len(tr_news_dataset['test']))
ml_sum_train_size = str(len(ml_sum_dataset['train']))
ml_sum_val_size = str(len(ml_sum_dataset['validation']))
ml_sum_test_size = str(len(ml_sum_dataset['test']))

print(
    "TR-News train size: {} val size: {} test size: {}".format(tr_news_train_size, tr_news_val_size, tr_news_test_size))
print("MLSum train size: {} val size: {} test size: {}".format(ml_sum_train_size, ml_sum_val_size, ml_sum_test_size))

tr_news_columns_to_remove = list(
    set(tr_news_dataset['train'].column_names) - set(['url', 'content', 'abstract', 'title', 'topic']))
ml_sum_columns_to_remove = list(
    set(ml_sum_dataset['train'].column_names) - set(['url', 'text', 'summary', 'title', 'topic']))

tr_news_train_df = tr_news_dataset['train'].to_pandas()
tr_news_train_df.rename(columns={'content': 'text', 'abstract': 'summary'}, inplace=True)
tr_news_train_df.drop(columns=tr_news_columns_to_remove, inplace=True)

tr_news_val_df = tr_news_dataset['validation'].to_pandas()
tr_news_val_df.rename(columns={'content': 'text', 'abstract': 'summary'}, inplace=True)
tr_news_val_df.drop(columns=tr_news_columns_to_remove, inplace=True)

tr_news_test_df = tr_news_dataset['test'].to_pandas()
tr_news_test_df.rename(columns={'content': 'text', 'abstract': 'summary'}, inplace=True)
tr_news_test_df.drop(columns=tr_news_columns_to_remove, inplace=True)

ml_sum_train_df = ml_sum_dataset['train'].to_pandas()
ml_sum_train_df.drop(columns=ml_sum_columns_to_remove, inplace=True)

ml_sum_val_df = ml_sum_dataset['validation'].to_pandas()
ml_sum_val_df.drop(columns=ml_sum_columns_to_remove, inplace=True)

ml_sum_test_df = ml_sum_dataset['test'].to_pandas()
ml_sum_test_df.drop(columns=ml_sum_columns_to_remove, inplace=True)

combined_train_df = pd.concat([tr_news_train_df, ml_sum_train_df])
combined_val_df = pd.concat([tr_news_val_df, ml_sum_val_df])
combined_test_df = pd.concat([tr_news_test_df, ml_sum_test_df])

combined_train_size = str(len(combined_train_df))
combined_val_size = str(len(combined_val_df))
combined_test_size = str(len(combined_test_df))
print("Combined dataset train size: {} val size: {} test size: {}".format(combined_train_size, combined_val_size,
                                                                          combined_test_size))

combined_train_df = combined_train_df.sample(frac=1).reset_index(drop=True)
combined_val_df = combined_val_df.sample(frac=1).reset_index(drop=True)
combined_test_df = combined_test_df.sample(frac=1).reset_index(drop=True)
print("Dataset shuffled.")

combined_train_df.to_csv( "data/combined_tr_raw/train.csv", index=False)
combined_val_df.to_csv( "data/combined_tr_raw/validation.csv", index=False)
combined_test_df.to_csv( "data/combined_tr_raw/test.csv", index=False)
print("Files saved.")