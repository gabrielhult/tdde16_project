from data_utils import load_data as ld
from preprocess import pre_process as pre, split_data as stt, vectorise_data as vec, sentiment_pre_process as stp
from bayes import multinomial, bernoulli
from svm import linear
from random_forest import forest
from dummy_baseline import dummy
import pandas as pd
import os
from tqdm import tqdm

def prepare_data(undersample=False):
    data = ld('reddit_posts/reddit_posts.csv')
    print("Values before new df but data loaded:", data['Political Lean'].value_counts())

    if undersample:
        print("Undersampling...")
        lowest_amount_of_speeches_by_lean = data['Political Lean'].value_counts().min()
        data_resampled = data.groupby('Political Lean').apply(lambda x: x.sample(n=lowest_amount_of_speeches_by_lean))
        print("Values after undersampling:", data_resampled['Political Lean'].value_counts())
    else:
        data_resampled = data

    title_text_csv = pd.DataFrame(columns=['Title_Text', 'Political Lean'])    
    for _, row in tqdm(data_resampled.iterrows(), total=len(data_resampled), desc="Processing data"):
        if pd.isnull(row['Text']):
            row['Text'] = ''
        title_text_csv.loc[len(title_text_csv)] = {'Title_Text': ' '.join([row['Title'], row['Text']]), 'Political Lean': row['Political Lean']}
    
    # Save the processed data as a new CSV file
    title_text_csv.to_csv('reddit_posts/reddit_posts_processed.csv', index=False)
    print("Values after data is prepared:", data_resampled['Political Lean'].value_counts())

def preprocess_data():
    data = ld('reddit_posts/reddit_posts_processed.csv')
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
    prepare_data(True)
    preprocess_data()
    split_dataset()
    vectoriser = vectorise()
    dummy(vectoriser)
    #multinomial(vectoriser)
    #bernoulli(vectoriser)
    linear(vectoriser)
    forest(vectoriser)
