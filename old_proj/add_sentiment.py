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
        max_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        if max_sentiment == 'pos':
            data.at[i, 'sentiment'] = 'Positive'
        elif max_sentiment == 'neg':
            data.at[i, 'sentiment'] = 'Negative'
        else:
            data.at[i, 'sentiment'] = 'Neutral'

    # Save data
    sentiment_file_name = 'sentiment_posts.csv'
    data.to_csv(sentiment_file_name, index=False)

    # Count the amount of neutral, negative, and positive sentiments
    neutral_count = len(data[data['sentiment'] == 'Neutral'])
    negative_count = len(data[data['sentiment'] == 'Negative'])
    positive_count = len(data[data['sentiment'] == 'Positive'])

    # Print the counts
    print("Neutral count:", neutral_count)
    print("Negative count:", negative_count)
    print("Positive count:", positive_count)

    # Plot sentiment
    # plt.plot(data['sentiment'])
    # plt.show()
    return sentiment_file_name