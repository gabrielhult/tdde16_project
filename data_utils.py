import pandas as pd
import pickle

def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()
    data = data.reset_index(drop=True)
    return data

def data_mapping():
    train_data = pd.read_csv('data/x_train_data.csv')
    test_data = pd.read_csv('data/x_test_data.csv')
    train_labels = pd.read_csv('data/y_train_data.csv')
    test_labels = pd.read_csv('data/y_test_data.csv')

    # Load vectorizer
    with open('data/vectoriser.pkl', 'rb') as file:
        vectoriser = pickle.load(file)

    # Vectorise data
    train_data = vectoriser.transform(train_data['Title_Text'])
    test_data = vectoriser.transform(test_data['Title_Text'])

    # Map class labels to numerical values
    class_mapping = {'Conservative': 0, 'Liberal': 1}
    train_labels_mapped = train_labels['Political Lean'].map(class_mapping)
    test_labels_mapped = test_labels['Political Lean'].map(class_mapping)
    return train_labels_mapped, test_labels_mapped, train_data, test_data
