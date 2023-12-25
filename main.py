from data_utils import load_data as ld
from preprocess import pre_process as pre, split_data as stt, vectorise_data as vec 
from models.bayes import multinomial, bernoulli
from models.svm import linear, rbf
import pandas as pd
import os


def preprocess_data():
    data = ld('reddit_posts/reddit_posts.csv')
    data['Title_Text'] = data['Title'] + ' ' + data['Text']
    data['Title_Text'] = pre(data['Title_Text'])
    print(data.head())
    data.to_csv('reddit_posts/reddit_posts_cleaned.csv', index=False)


def split_dataset():
    train_csv, test_csv = stt()
    train_csv_path = os.path.join("data", train_csv)
    test_csv_path = os.path.join("data", test_csv)
    train_data = pd.read_csv(train_csv_path)
    test_data = pd.read_csv(test_csv_path)
    print(train_data.head())
    print(test_data.head())


def vectorise():
    vectoriser = vec()
    idf_tuples = sorted(zip(vectoriser.idf_, vectoriser.get_feature_names_out()))
    terms = [term for _, term in idf_tuples]
    print(f"Terms with the lowest idf:\n{terms[:10]}\n")
    print(f"Terms with the highest idf:\n{terms[-10:]}")
    print(vectoriser.get_feature_names_out())


def bayes():
    multinomial()
    bernoulli()

def svm():
    linear()
    rbf()


if __name__ == "__main__":
    #preprocess_data()
    split_dataset()
    vectorise()
    bayes()
    #svm()
