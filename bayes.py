import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from textblob import TextBlob
from data_utils import data_mapping as dm
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def multinomial(vectoriser):
    train_labels_mapped, test_labels_mapped, train_data, test_data, data = dm()

    leans = sorted(train_labels_mapped.unique())
    # Train model
    model = MultinomialNB().fit(train_data, train_labels_mapped)

    # Print out the most decisive words for classifying as either "Liberal" or "Conservative"
    liberal_top_words, conservative_top_words = decisive_words(model, vectoriser, 'multinomial')

    # Identify rows with top words for each class
    liberal_rows = data[data['Title_Text'].str.contains('|'.join(liberal_top_words), case=False)]
    conservative_rows = data[data['Title_Text'].str.contains('|'.join(conservative_top_words), case=False)]

    # Extract sentences with top words
    liberal_sentences = liberal_rows['Title_Text'].tolist()
    conservative_sentences = conservative_rows['Title_Text'].tolist()

    # Perform sentiment analysis using TextBlob
    def get_sentiment_textblob(sentence):
        analysis = TextBlob(sentence)
        return analysis.sentiment.polarity
    
    # Perform sentiment analysis using VADER
    def get_sentiment_vader(sentence):
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = analyzer.polarity_scores(sentence)
        return sentiment_scores['compound']

    # Analyze sentiment for liberal sentences
    liberal_sentiments_blob = [get_sentiment_textblob(sentence) for sentence in liberal_sentences]
    liberal_sentiments_vader = [get_sentiment_vader(sentence) for sentence in liberal_sentences]


    # Analyze sentiment for conservative sentences
    conservative_sentiments_blob = [get_sentiment_textblob(sentence) for sentence in conservative_sentences]
    conservative_sentiments_vader = [get_sentiment_vader(sentence) for sentence in conservative_sentences]

    # Calculate average sentiment
    average_liberal_sentiment_blob = sum(liberal_sentiments_blob) / len(liberal_sentiments_blob)
    average_liberal_sentiment_vader = sum(liberal_sentiments_vader) / len(liberal_sentiments_vader)
    average_conservative_sentiment_blob = sum(conservative_sentiments_blob) / len(conservative_sentiments_blob)
    average_conservative_sentiment_vader = sum(conservative_sentiments_vader) / len(conservative_sentiments_vader)

    # Print average sentiment
    print("Average Sentiment for Liberal Sentences (TextBlob):", average_liberal_sentiment_blob)
    print("Average Sentiment for Liberal Sentences (VADER):", average_liberal_sentiment_vader)
    print("Average Sentiment for Conservative Sentences (TextBlob):", average_conservative_sentiment_blob)
    print("Average Sentiment for Conservative Sentences (VADER):", average_conservative_sentiment_vader)

    # Predict
    predicted = model.predict(test_data)

    # Evaluate
    report = classification_report(test_labels_mapped, predicted, target_names=leans, zero_division=np.nan)
    print(f"Multinomial Naive Bayes Classifier:\n", report)

    # Plot confusion matrix
    cm = confusion_matrix(test_labels_mapped, predicted)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Multinomial Naive Bayes Classifier')
    plt.savefig('/home/gult/tdde16_project/graphs/multi_confusion_matrix.pdf')

    # Save model
    model_file_name = 'multinomial_bayes.sav'
    with open(os.path.join('models', model_file_name), 'wb') as file:
        pickle.dump(model, file)


def bernoulli(vectoriser):
    train_labels_mapped, test_labels_mapped, train_data, test_data, data = dm()

    leans = sorted(train_labels_mapped.unique())

    # Train model
    model = BernoulliNB().fit(train_data, train_labels_mapped)

    # Print out the most decisive words for classifying as either "Liberal" or "Conservative"
    liberal_top_words, conservative_top_words = decisive_words(model, vectoriser, 'bernoulli')

    # Identify rows with top words for each class
    liberal_rows = data[data['Title_Text'].str.contains('|'.join(liberal_top_words), case=False)]
    conservative_rows = data[data['Title_Text'].str.contains('|'.join(conservative_top_words), case=False)]

    # Extract sentences with top words
    liberal_sentences = liberal_rows['Title_Text'].tolist()
    conservative_sentences = conservative_rows['Title_Text'].tolist()

    # Perform sentiment analysis using TextBlob
    def get_sentiment_textblob(sentence):
        analysis = TextBlob(sentence)
        return analysis.sentiment.polarity
    
    # Perform sentiment analysis using VADER
    def get_sentiment_vader(sentence):
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = analyzer.polarity_scores(sentence)
        return sentiment_scores['compound']

    # Analyze sentiment for liberal sentences
    liberal_sentiments_blob = [get_sentiment_textblob(sentence) for sentence in liberal_sentences]
    liberal_sentiments_vader = [get_sentiment_vader(sentence) for sentence in liberal_sentences]

    # Analyze sentiment for conservative sentences
    conservative_sentiments_blob = [get_sentiment_textblob(sentence) for sentence in conservative_sentences]
    conservative_sentiments_vader = [get_sentiment_vader(sentence) for sentence in conservative_sentences]

    # Calculate average sentiment
    average_liberal_sentiment_blob = sum(liberal_sentiments_blob) / len(liberal_sentiments_blob)
    average_liberal_sentiment_vader = sum(liberal_sentiments_vader) / len(liberal_sentiments_vader)
    average_conservative_sentiment_blob = sum(conservative_sentiments_blob) / len(conservative_sentiments_blob)
    average_conservative_sentiment_vader = sum(conservative_sentiments_vader) / len(conservative_sentiments_vader)

    # Print average sentiment
    print("Average Sentiment for Liberal Sentences (TextBlob):", average_liberal_sentiment_blob)
    print("Average Sentiment for Liberal Sentences (VADER):", average_liberal_sentiment_vader)
    print("Average Sentiment for Conservative Sentences (TextBlob):", average_conservative_sentiment_blob)
    print("Average Sentiment for Conservative Sentences (VADER):", average_conservative_sentiment_vader)

    # Predict
    predicted = model.predict(test_data)

    # Evaluate
    report = classification_report(test_labels_mapped, predicted, target_names=leans, zero_division=np.nan)
    print(f"Bernoulli Naive Bayes Classifier:\n", report)


    # Plot confusion matrix
    cm = confusion_matrix(test_labels_mapped, predicted)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Bernoulli Naive Bayes Classifier')
    plt.savefig('/home/gult/tdde16_project/graphs/bernoulli_confusion_matrix.pdf')

    # Save model
    model_file_name = 'bernoulli_bayes.sav'
    with open(os.path.join('models', model_file_name), 'wb') as file:
        pickle.dump(model, file)

def decisive_words(model, vectoriser, model_name):
    # Get feature log probabilities
    feature_log_probs = model.feature_log_prob_

    # Top words for each class
    liberal_top_words = [vectoriser.get_feature_names_out()[i] for i in feature_log_probs[0].argsort()[-10:][::-1]]
    conservative_top_words = [vectoriser.get_feature_names_out()[i] for i in feature_log_probs[1].argsort()[-10:][::-1]]


    # Print or visualize the top words for each class
    print("Top words for Liberal:", liberal_top_words)
    print("Top words for Conservative:", conservative_top_words)


    # # Save the top words with their log probability differences for liberal to a text file
    output_file_path = os.path.join('top_words', f"{model_name}_liberal_words.txt")
    with open(output_file_path, "w") as file:
        file.write(f"Top 10 words for classifying as 'liberal':\n")
        for word in liberal_top_words:
            file.write(f"Word: {word}\n")

    # Save the top words with their log probability differences for conservative to a text file
    output_file_path = os.path.join('top_words', f"{model_name}_conservative_words.txt")
    with open(output_file_path, "w") as file:
        file.write(f"Top 10 words for classifying as 'conservative':\n")
        for word in conservative_top_words:
            file.write(f"Word: {word}\n")

    return liberal_top_words, conservative_top_words
