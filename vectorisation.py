
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import pandas as pd
import os
import pickle
from tqdm import tqdm

def vectorise_data():
    # Load data
    train_data = pd.read_csv('data/train_data.csv')
    test_data = pd.read_csv('data/test_data.csv')
    print(train_data.head())
    print(test_data.head())

    # Vectorise data
    vectoriser = TfidfVectorizer()
    train_vectors = vectoriser.fit_transform(train_data['Text'])
    test_vectors = vectoriser.transform(test_data['Text'])

    # Save vectorised data
    train_vectors_file_name = 'train_vectors.npz'
    test_vectors_file_name = 'test_vectors.npz'
    with tqdm(total=2) as pbar:
        save_npz(os.path.join('data', train_vectors_file_name), train_vectors)
        pbar.update(1)
        save_npz(os.path.join('data', test_vectors_file_name), test_vectors)
        pbar.update(1)

    # Save vectoriser
    vectoriser_file_name = 'vectoriser.pkl'
    with open(os.path.join('data', vectoriser_file_name), 'wb') as f:
        pickle.dump(vectoriser, f)