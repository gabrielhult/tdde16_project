import spacy
from tqdm import tqdm
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import pickle
from data_utils import load_data as ld
import re
import contractions
import numpy as np

nlp = spacy.load("en_core_web_lg")

def sentiment_process_text(text):
    # Expand contractions
    text = ' '.join([contractions.fix(word) for word in text.split()])
    
    # Remove some punctuations (not ones that can be used in emoticons)
    text = re.sub(r'[^\w\s\:\)\(\;\-\=\>\<\[\]\{\}\|\_\+\!\@\#\$\%\^\&\*]', '', text)
    
    # Lemmatize, remove stop words, and convert to lowercase
    return ' '.join([token.lemma_.lower() for token in nlp(text) if not token.is_stop])

def sentiment_pre_process(col_text):
    processed_texts = []

    # Use tqdm to create a progress bar
    for text in tqdm(col_text, desc="Processing text for sentiment prediction", unit="text"):
        processed_text = process_text(text)
        processed_texts.append(processed_text)
    
    return processed_texts

def process_text(text):
    # Expand contractions
    text = ' '.join([contractions.fix(word) for word in text.split()])
    
    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    
    # Lemmatize, remove stop words, and convert to lowercase
    return ' '.join([token.lemma_.lower() for token in nlp(text) if token.is_alpha and not token.is_stop])

def pre_process(col_text):
    processed_texts = []

    # Use tqdm to create a progress bar
    for text in tqdm(col_text, desc="Processing text", unit="text"):
        processed_text = process_text(text)
        processed_texts.append(processed_text)
    
    return processed_texts


def split_data():
    # Load data
    data = ld('reddit_posts/reddit_posts_cleaned.csv')

    with tqdm(total=5) as pbar:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data['Title_Text'], data['Political Lean'], test_size=0.2, random_state=42)

        pbar.update(1)

        # Create 'data' directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
            
        # Save data
        train_file_name = f'x_train_data.csv'
        test_file_name = f'x_test_data.csv'
        X_train.to_csv(os.path.join('data', train_file_name), index=False)
        pbar.update(1)
        y_train.to_csv(os.path.join('data', train_file_name.replace('x_train', 'y_train')), index=False)
        pbar.update(1)
        X_test.to_csv(os.path.join('data', test_file_name), index=False)
        pbar.update(1)
        y_test.to_csv(os.path.join('data', test_file_name.replace('x_test', 'y_test')), index=False)
        pbar.update(1)

    return train_file_name, test_file_name


def vectorise_data():
    # Load data
    train_data = pd.read_csv('data/x_train_data.csv')
    test_data = pd.read_csv('data/x_test_data.csv')
    print(train_data.head())
    print(test_data.head())

    # Replace NaN values with empty strings
    train_data['Title_Text'] = train_data['Title_Text'].fillna('')
    test_data['Title_Text'] = test_data['Title_Text'].fillna('')

    # Vectorise data
    vectoriser = TfidfVectorizer()
    train_vectors = vectoriser.fit_transform(train_data['Title_Text'])
    test_vectors = vectoriser.transform(test_data['Title_Text'])

    # Save vectorised data
    train_vectors_file_name = 'train_vectors.npz'
    test_vectors_file_name = 'test_vectors.npz'
    with tqdm(total=2) as pbar:
        save_npz(os.path.join('data', train_vectors_file_name), train_vectors)
        pbar.update(1)
        save_npz(os.path.join('data', test_vectors_file_name), test_vectors)
        pbar.update(1)
    vectoriser_file_name = 'vectoriser.pkl'
    with open(os.path.join('data', vectoriser_file_name), 'wb') as file:
        pickle.dump(vectoriser, file)

    return vectoriser
