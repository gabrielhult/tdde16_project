import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def split_data():
    # Load data
    data = pd.read_csv('reddit_posts/reddit_posts_cleaned.csv')
    print(data.head())
    data = data.dropna()
    data = data.reset_index(drop=True)

    # Split data into training and testing sets
    train_data = data.sample(frac=0.8, random_state=0)
    test_data = data.drop(train_data.index)

    # Create 'data' directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Save data with progress bar
    train_file_name = 'train_data.csv'
    test_file_name = 'test_data.csv'
    with tqdm(total=2) as pbar:
        train_data.to_csv(os.path.join('data', train_file_name), index=False)
        pbar.update(1)
        test_data.to_csv(os.path.join('data', test_file_name), index=False)
        pbar.update(1)

    # Plot sentiment
    # plt.plot(data['sentiment'])
    # plt.show()
    return train_file_name, test_file_name