import preprocess as pre
import split_train_test as stt
import pandas as pd
import os


def preprocess_data():
    data = pd.read_csv('reddit_posts/reddit_posts.csv')
    data = data.dropna()
    data = data.reset_index(drop=True)
    data['Title'] = pre.pre_process(data['Title'])
    data['Text'] = pre.pre_process(data['Text'])
    print(data.head())
    data.to_csv('reddit_posts/reddit_posts_cleaned.csv', index=False)


def split_dataset():
    train_csv, test_csv = stt.split_data()
    train_csv_path = os.path.join("data", train_csv)
    test_csv_path = os.path.join("data", test_csv)
    train_data = pd.read_csv(train_csv_path)
    test_data = pd.read_csv(test_csv_path)
    print(train_data.head())
    print(test_data.head())



if __name__ == "__main__":
    #preprocess_data()
    split_dataset()
