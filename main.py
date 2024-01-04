from data_utils import load_data as ld
from preprocess import pre_process as pre, split_data as stt, vectorise_data as vec, sentiment_pre_process as stp
from bayes import multinomial, bernoulli
from svm import linear
from random_forest import forest
from dummy_baseline import dummy
import pandas as pd
import os
from imblearn.under_sampling import RandomUnderSampler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from tqdm import tqdm

def prepare_data(undersample=False):
    data = ld('reddit_posts/reddit_posts.csv')

    if undersample:
        print("Undersampling...")
        # Undersample the majority class
        X = data.drop('Political Lean', axis=1)
        y = data['Political Lean']
        sampler = RandomUnderSampler(random_state=42, sampling_strategy='majority')
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Combine resampled features and labels
        data_resampled = pd.concat([X_resampled, y_resampled], axis=1)
    else:
        data_resampled = data

    title_text_csv = pd.DataFrame(columns=['Title_Text', 'Political Lean'])    
    for _, row in tqdm(data_resampled.iterrows(), total=len(data_resampled), desc="Processing data"):
        title_text_csv.loc[len(title_text_csv)] = {'Title_Text': ' '.join([row['Title'], row['Text']]), 'Political Lean': row['Political Lean']}
    title_text_csv.to_csv('reddit_posts/reddit_posts_title_text.csv', index=False)

def sentiment_prediction():
    data = ld('reddit_posts/reddit_posts_title_text.csv')
    data['Title_Text'] = stp(data['Title_Text'])
    data['Sentiment_VADER'] = data['Title_Text'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])
    data['Sentiment_TextBlob'] = data['Title_Text'].apply(lambda x: TextBlob(x).sentiment.polarity)
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


if __name__ == "__main__":
    #prepare_data(False)
    #sentiment_prediction() # Only for sentiment prediction to further analysis
    #preprocess_data()
    #split_dataset()
    vectoriser = vectorise()
    dummy(vectoriser)
    multinomial(vectoriser)
    bernoulli(vectoriser)
    linear(vectoriser)
    forest(vectoriser)
