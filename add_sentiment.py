from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def add_sentiment():
    # Load data
    data = pd.read_csv('reddit_posts.csv')
    print(data.head())
    data = data.dropna()
    data = data.reset_index(drop=True)

    # Create sentiment column
    data['sentiment'] = np.nan

    analyser = SentimentIntensityAnalyzer()

    # Create sentiment column
    for i in range(len(data)):
        data['sentiment'][i] = analyser.polarity_scores(data['Title'][i])

    sentiment_file_name = 'data_sentiment.csv'
    # Save data
    data.to_csv(sentiment_file_name, index=False)

    # Plot sentiment
    # plt.plot(data['sentiment'])
    # plt.show()
    return sentiment_file_name