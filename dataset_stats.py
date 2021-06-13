import re

import pandas as pd
from nltk import sent_tokenize, word_tokenize


def read_dataset(train_file, val_file, test_file):
    print("Started reading dataset!")
    news_df_train = pd.read_csv(train_file)
    print("Train size: ", str(len(news_df_train)))
    news_df_val = pd.read_csv(val_file)
    print("Validation size: ", str(len(news_df_val)))
    news_df_test = pd.read_csv(test_file)
    print("Test size: ", str(len(news_df_test)))

    print("Finished reading dataset!")

    return pd.concat([news_df_train, news_df_val, news_df_test])


def lower_tr(text):
    text = re.sub(r'İ', 'i', text)
    return re.sub(r'I', 'ı', text).lower().strip()


def text_count(text):
    sent_count = 0
    word_count = 0

    for sent in sent_tokenize(text):
        word_count += len(word_tokenize(sent))
        sent_count += 1
    return [sent_count, word_count]


def tokenize(text):
    tokens = []
    for sent in sent_tokenize(text):
        tokens.extend(word_tokenize(sent))
    return tokens


def calc_lengths():
    datasets = ["tr_news_raw", "ml_sum_tr_raw", "combined_tr_raw"]

    for dataset in datasets:
        train_file = "data/" + dataset + "/" + "train.csv"
        validation_file = "data/" + dataset + "/" + "validation.csv"
        test_file = "data/" + dataset + "/" + "test.csv"

        df = read_dataset(train_file, validation_file, test_file)

        title_column = "title"
        if dataset == "tr_news_raw":
            source_column = "content"
            target_column = "abstract"
        else:
            source_column = "text"
            target_column = "summary"

        df[source_column] = df[source_column].apply(lower_tr)
        df[target_column] = df[target_column].apply(lower_tr)
        df[title_column] = df[title_column].apply(lower_tr)

        df[[target_column + "_sent_count", target_column + "_word_count"]] = df.apply(
            lambda row: pd.Series(text_count(row[target_column], )), axis=1, )
        df[[source_column + "_sent_count", source_column + "_word_count"]] = df.apply(
            lambda row: pd.Series(text_count(row[source_column], )), axis=1, )
        df[[title_column + "_sent_count", title_column + "_word_count"]] = df.apply(
            lambda row: pd.Series(text_count(row[title_column], )), axis=1, )

        print("Dataset: {} \t Avg word count {} : {} \t {} : {} ".format(dataset, source_column,
                                                                         df[source_column + "_word_count"].mean(),
                                                                         target_column,
                                                                         df[target_column + "_word_count"].mean()))
        print("Dataset: {} \t Avg sentence count {} : {} \t {} : {}".format(dataset, source_column,
                                                                            df[source_column + "_sent_count"].mean(),
                                                                            target_column,
                                                                            df[target_column + "_sent_count"].mean()))

        print("Dataset: {} \t Avg word count {} : {}".format(dataset, title_column,df[title_column + "_word_count"].mean()))


def calc_vocab():
    datasets = ["tr_news_raw", "ml_sum_tr_raw", "combined_tr_raw"]

    for dataset in datasets:
        train_file = "data/" + dataset + "/" + "train.csv"
        validation_file = "data/" + dataset + "/" + "validation.csv"
        test_file = "data/" + dataset + "/" + "test.csv"

        df = read_dataset(train_file, validation_file, test_file)

        content_vocab_count = set()
        abstract_vocab_count = set()
        title_vocab_count = set()
        overall_vocab_count = set()

        title_column = "title"
        if dataset == "tr_news_raw":
            source_column = "content"
            target_column = "abstract"
        else:
            source_column = "text"
            target_column = "summary"

        df[source_column] = df[source_column].apply(lower_tr)
        df[target_column] = df[target_column].apply(lower_tr)
        df[title_column] = df[title_column].apply(lower_tr)

        for idx, row in df.iterrows():
            content_vocab_count.update(tokenize(row[source_column]))
            abstract_vocab_count.update(tokenize(row[target_column]))
            title_vocab_count.update(tokenize(row[title_column]))

        overall_vocab_count.update(content_vocab_count)
        overall_vocab_count.update(abstract_vocab_count)
        overall_vocab_count.update(title_vocab_count)

        print("Dataset: " + dataset)
        print("Content vocab size: " + str(len(content_vocab_count)))
        print("Abstract vocab size: " + str(len(abstract_vocab_count)))
        print("Title vocab size: " + str(len(title_vocab_count)))
        print("Overall vocab size: " + str(len(overall_vocab_count)))

if __name__ == "__main__":
    calc_lengths()
    calc_vocab()
