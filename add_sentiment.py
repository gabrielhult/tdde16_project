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

    # Create sentiment column if it doesn't exist
    if 'sentiment' not in data.columns:
        data['sentiment'] = np.nan

    analyser = SentimentIntensityAnalyzer()

    # Add sentiment data to sentiment column
    for i in range(len(data)):
        sentiment_scores = analyser.polarity_scores(data['Title'][i])
        data.at[i, 'sentiment'] = str(sentiment_scores)  # Convert dictionary to string

    # Save data
    sentiment_file_name = 'sentiment_posts.csv'
    data.to_csv(sentiment_file_name, index=False)

    # Plot sentiment
    # plt.plot(data['sentiment'])
    # plt.show()
    return sentiment_file_name