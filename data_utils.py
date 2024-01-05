import pandas as pd
import pickle
import numpy as np


def load_data(file_path):
    data = pd.read_csv(file_path)
    print("Values with 0 mod:", data['Political Lean'].value_counts())
    return data


def data_mapping():
    train_title_text = pd.read_csv('data/x_train_data.csv')
    test_title_text = pd.read_csv('data/x_test_data.csv')
    train_labels = pd.read_csv('data/y_train_data.csv')
    test_labels = pd.read_csv('data/y_test_data.csv')

    # Load vectorizer
    with open('data/vectoriser.pkl', 'rb') as file:
        vectoriser = pickle.load(file)

    # Vectorise data
    train_title_text = vectoriser.transform(train_title_text['Title_Text'])
    test_title_text = vectoriser.transform(test_title_text['Title_Text'])

    # Map class labels to numerical values
    #class_mapping = {'Conservative': 0, 'Liberal': 1}
    train_labels_mapped = train_labels['Political Lean']
    test_labels_mapped = test_labels['Political Lean']
    return train_labels_mapped, test_labels_mapped, train_title_text, test_title_text
