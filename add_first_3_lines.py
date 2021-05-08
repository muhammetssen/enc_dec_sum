from datasets import load_dataset
from nltk import sent_tokenize


def get_first_lines(text, n=3):
    return " ".join(sent_tokenize(text)[:n])


# TR-News dataset variables
tr_news_data_files = dict()
tr_news_data_files["train"] = "data/tr_news_raw/train.csv"
tr_news_data_files["validation"] = "data/tr_news_raw/validation.csv"
tr_news_data_files["test"] = "data/tr_news_raw/test.csv"

# combined tr dataset variables
combined_tr_data_files = dict()
combined_tr_data_files["train"] = "data/combined_tr_raw/train.csv"
combined_tr_data_files["validation"] = "data/combined_tr_raw/validation.csv"
combined_tr_data_files["test"] = "data/combined_tr_raw/test.csv"

tr_news_dataset = load_dataset("csv", data_files=tr_news_data_files)
ml_sum_dataset = load_dataset("mlsum", "tu")
combined_tr_dataset = load_dataset("csv", data_files=combined_tr_data_files)

tr_news_train_size = str(len(tr_news_dataset['train']))
tr_news_val_size = str(len(tr_news_dataset['validation']))
tr_news_test_size = str(len(tr_news_dataset['test']))

ml_sum_train_size = str(len(ml_sum_dataset['train']))
ml_sum_val_size = str(len(ml_sum_dataset['validation']))
ml_sum_test_size = str(len(ml_sum_dataset['test']))

combined_tr_train_size = str(len(combined_tr_dataset['train']))
combined_tr_val_size = str(len(combined_tr_dataset['validation']))
combined_tr_test_size = str(len(combined_tr_dataset['test']))

print(
    "TR-News train size: {} val size: {} test size: {}".format(tr_news_train_size, tr_news_val_size, tr_news_test_size))
print("MLSum train size: {} val size: {} test size: {}".format(ml_sum_train_size, ml_sum_val_size, ml_sum_test_size))
print("Combined TR train size: {} val size: {} test size: {}".format(combined_tr_train_size, combined_tr_val_size,
                                                                     combined_tr_test_size))

tr_news_train_df = tr_news_dataset['train'].to_pandas()
tr_news_val_df = tr_news_dataset['validation'].to_pandas()
tr_news_test_df = tr_news_dataset['test'].to_pandas()
ml_sum_train_df = ml_sum_dataset['train'].to_pandas()
ml_sum_val_df = ml_sum_dataset['validation'].to_pandas()
ml_sum_test_df = ml_sum_dataset['test'].to_pandas()
combined_tr_train_df = combined_tr_dataset['train'].to_pandas()
combined_tr_val_df = combined_tr_dataset['validation'].to_pandas()
combined_tr_test_df = combined_tr_dataset['test'].to_pandas()

tr_news_train_df['first3'] = tr_news_train_df['content'].apply(get_first_lines)
tr_news_val_df['first3'] = tr_news_val_df['content'].apply(get_first_lines)
tr_news_test_df['first3'] = tr_news_test_df['content'].apply(get_first_lines)
ml_sum_train_df['first3'] = ml_sum_train_df['text'].apply(get_first_lines)
ml_sum_val_df['first3'] = ml_sum_val_df['text'].apply(get_first_lines)
ml_sum_test_df['first3'] = ml_sum_test_df['text'].apply(get_first_lines)
combined_tr_train_df['first3'] = combined_tr_train_df['text'].apply(get_first_lines)
combined_tr_val_df['first3'] = combined_tr_val_df['text'].apply(get_first_lines)
combined_tr_test_df['first3'] = combined_tr_test_df['text'].apply(get_first_lines)

print(
    "TR-News train size: {} val size: {} test size: {}".format(tr_news_train_size, tr_news_val_size, tr_news_test_size))
print("MLSum train size: {} val size: {} test size: {}".format(ml_sum_train_size, ml_sum_val_size, ml_sum_test_size))
print("Combined TR train size: {} val size: {} test size: {}".format(combined_tr_train_size, combined_tr_val_size,
                                                                     combined_tr_test_size))

tr_news_train_df.to_csv( "data/tr_news_raw/train.csv", index=False)
tr_news_val_df.to_csv( "data/tr_news_raw/validation.csv", index=False)
tr_news_test_df.to_csv( "data/tr_news_raw/test.csv", index=False)

ml_sum_train_df.to_csv( "data/ml_sum_tr_raw/train.csv", index=False)
ml_sum_val_df.to_csv( "data/ml_sum_tr_raw/validation.csv", index=False)
ml_sum_test_df.to_csv( "data/ml_sum_tr_raw/test.csv", index=False)

combined_tr_train_df.to_csv( "data/combined_tr_raw/train.csv", index=False)
combined_tr_val_df.to_csv( "data/combined_tr_raw/validation.csv", index=False)
combined_tr_test_df.to_csv( "data/combined_tr_raw/test.csv", index=False)
print("Files saved.")