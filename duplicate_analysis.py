import re

import pandas as pd


def lower_tr(text):
    text = re.sub(r'İ', 'i', text)
    return re.sub(r'I', 'ı', text).lower().strip()


if __name__ == "__main__":
    TOP_N = 10
    for dataset in ["data/ml_sum_tr_raw/", "data/tr_news_raw/"]:
        train_df = pd.read_csv(dataset + "train.csv")
        val_df = pd.read_csv(dataset + "validation.csv")
        test_df = pd.read_csv(dataset + "test.csv")
        df = pd.concat([train_df, val_df, test_df])

        text_field = None
        abstract_field = None
        title_field = "title"
        if "tr_news" in dataset:
            text_field = "content"
            abstract_field = "abstract"
        else:
            text_field = "text"
            abstract_field = "summary"

        df[text_field] = df[text_field].apply(lower_tr)
        df[title_field] = df[title_field].apply(lower_tr)
        df[abstract_field] = df[abstract_field].apply(lower_tr)

        for field in [text_field, abstract_field, title_field]:
            duplicates_df = df[df.duplicated([field], keep=False)]
            total_unique_dups = len(duplicates_df)
            duplicate_groups = duplicates_df[[field]].groupby(field)
            duplicate_counts_of_groups = duplicate_groups.size().values
            top_dups_in_group = duplicate_counts_of_groups[duplicate_counts_of_groups.argsort()[-TOP_N:][::-1]]

            print("Dataset: {} \t Field: {}".format(dataset, field))
            print("Number of duplicate groups:", str(len(duplicate_groups)))
            print("Total number of duplicates:", str(total_unique_dups))
            print("Top 10 max number of duplicates groups:", str(top_dups_in_group))

        duplicates_df = df[df.duplicated([text_field, abstract_field, title_field], keep=False)]
        total_unique_dups = len(duplicates_df)
        duplicate_groups = duplicates_df[[text_field]].groupby(text_field)
        duplicate_counts_of_groups = duplicate_groups.size().values
        top_dups_in_group = duplicate_counts_of_groups[duplicate_counts_of_groups.argsort()[-10:][::-1]]

        print("Dataset: {} \t Field: {}".format(dataset, "Document level"))
        print("Number of duplicate groups:", str(len(duplicate_groups)))
        print("Total number of duplicates:", str(total_unique_dups))
        print("Top 10 max number of duplicates groups:", str(top_dups_in_group))
