from data_utils import load_data as ld
from preprocess import pre_process as pre, split_data as stt, vectorise_data as vec, sentiment_pre_process as stp
from models.bayes import multinomial, bernoulli
from models.svm import linear
from models.random_forest import forest
import pandas as pd
import os
import vaderSentiment

def prepare_data():
    # TODO: Fix so the Title_Text column is correctly created
    data = ld('reddit_posts/reddit_posts.csv')
    data['Title_Text'] = data['Title'] + ' ' + data['Text']
    data.to_csv('reddit_posts/reddit_posts_title_text.csv', index=False)

def sentiment_prediction():
    data = ld('reddit_posts/reddit_posts_title_text.csv')
    data['Title_Text'] = stp(data['Title_Text'])
    print(data.head())
    vaderSentiment_model = vaderSentiment.SentimentIntensityAnalyzer()
    for row in data.itertuples():
        data["Sentiment"] = vaderSentiment_model.polarity_scores(row.Title_Text)
        print("{:-<65} {}".format(row.Title_Text, str(data["Sentiment"])))
    data.to_csv('reddit_posts/reddit_posts_sentiment.csv', index=False)

def preprocess_data():
    data = ld('reddit_posts/reddit_posts_title_text.csv')
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
    return vectoriser


def bayes(vectoriser):
    multinomial(vectoriser)
    bernoulli(vectoriser)

def svm(vectoriser):
    linear(vectoriser)

def random_forest(vectoriser):
    forest(vectoriser)


if __name__ == "__main__":
    prepare_data()
    sentiment_prediction()
    #preprocess_data()
    #split_dataset()
    #vectoriser = vectorise()
    #bayes(vectoriser)
    #svm(vectoriser)
    #random_forest(vectoriser)
