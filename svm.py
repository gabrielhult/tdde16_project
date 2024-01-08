import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data_utils import data_mapping as dm
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def linear(vectoriser):
    train_labels_mapped, test_labels_mapped, train_data, test_data, data = dm()

    leans = sorted(train_labels_mapped.unique())

    # Train model
    model = LinearSVC().fit(train_data, train_labels_mapped)

    # Print out the most decisive words for classifying as either "Liberal" or "Conservative"
    liberal_top_words, conservative_top_words = decisive_words(model, vectoriser, 'linear')

    sentiment_prediction(data, liberal_top_words, conservative_top_words, 'linear')

    # Predict
    predicted = model.predict(test_data)

    # Evaluate
    report = classification_report(test_labels_mapped, predicted, target_names=leans, zero_division=np.nan)
    print(f"Linear SVM Classifier:\n", report)

    # Plot confusion matrix
    cm = confusion_matrix(test_labels_mapped, predicted)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Linear SVM Classifier')
    plt.savefig('/home/gult/tdde16_project/graphs/linear_confusion_matrix.pdf')

    # Save model
    model_file_name = 'linear_svm.sav'
    with open(os.path.join('models', model_file_name), 'wb') as file:
        pickle.dump(model, file)


def decisive_words(model, vectoriser, model_name):
    # Get feature coefficients
    feature_coeffs = model.coef_

    print(feature_coeffs.shape, feature_coeffs)

    # Top words for each class
    liberal_top_words = [vectoriser.get_feature_names_out()[i] for i in feature_coeffs[0].argsort()[-10:][::-1]]
    conservative_top_words = [vectoriser.get_feature_names_out()[i] for i in feature_coeffs[0].argsort()[-10:][::-1]]

    # Print or visualize the top words for each class
    print("Top words for Liberal:", liberal_top_words)
    print("Top words for Conservative:", conservative_top_words)

    # Save the top words with their coefficients for liberal to a text file
    output_file_path = os.path.join('top_words', f"{model_name}_liberal_words.txt")
    with open(output_file_path, "w") as file:
        file.write(f"Top 10 words for classifying as 'liberal':\n")
        for word in liberal_top_words:
            file.write(f"Word: {word}\n")

    # Save the top words with their coefficients for conservative to a text file
    output_file_path = os.path.join('top_words', f"{model_name}_conservative_words.txt")
    with open(output_file_path, "w") as file:
        file.write(f"Top 10 words for classifying as 'conservative':\n")
        for word in conservative_top_words:
            file.write(f"Word: {word}\n")

    return liberal_top_words, conservative_top_words


def sentiment_prediction(data, liberal_top_words, conservative_top_words, model_name):
    # Identify rows with top words for each class
    liberal_rows = data[data['Title_Text'].str.contains('|'.join(liberal_top_words), case=False)]
    conservative_rows = data[data['Title_Text'].str.contains('|'.join(conservative_top_words), case=False)]

    # Extract sentences with top words
    liberal_sentences = liberal_rows['Title_Text'].tolist()
    conservative_sentences = conservative_rows['Title_Text'].tolist()

    # Perform sentiment analysis using TextBlob
    liberal_sentiments_blob = [TextBlob(sentence).sentiment.polarity for sentence in liberal_sentences]
    conservative_sentiments_blob = [TextBlob(sentence).sentiment.polarity for sentence in conservative_sentences]

    # Perform sentiment analysis using VADER
    analyzer = SentimentIntensityAnalyzer()
    liberal_sentiments_vader = [analyzer.polarity_scores(sentence)['compound'] for sentence in liberal_sentences]
    conservative_sentiments_vader = [analyzer.polarity_scores(sentence)['compound'] for sentence in conservative_sentences]

    # Calculate average sentiment
    average_liberal_sentiment_blob = sum(liberal_sentiments_blob) / len(liberal_sentiments_blob)
    average_liberal_sentiment_vader = sum(liberal_sentiments_vader) / len(liberal_sentiments_vader)
    average_conservative_sentiment_blob = sum(conservative_sentiments_blob) / len(conservative_sentiments_blob)
    average_conservative_sentiment_vader = sum(conservative_sentiments_vader) / len(conservative_sentiments_vader)

    # Calculate maximum and minimum sentiment scores
    max_liberal_sentiment_blob = max(liberal_sentiments_blob)
    min_liberal_sentiment_blob = min(liberal_sentiments_blob)
    max_liberal_sentiment_vader = max(liberal_sentiments_vader)
    min_liberal_sentiment_vader = min(liberal_sentiments_vader)
    max_conservative_sentiment_blob = max(conservative_sentiments_blob)
    min_conservative_sentiment_blob = min(conservative_sentiments_blob)
    max_conservative_sentiment_vader = max(conservative_sentiments_vader)
    min_conservative_sentiment_vader = min(conservative_sentiments_vader)

    # Initialize counters
    sentiment_counts = {'Negative': 0, 'Neutral': 0, 'Positive': 0}

    # Iterate over liberal sentiments
    sentiments = liberal_sentiments_vader + conservative_sentiments_vader
    sentiment_labels = ['Negative' if sentiment < 0 else 'Neutral' if sentiment == 0 else 'Positive' for sentiment in
                        sentiments]

    # Count the sentiments
    for label in sentiment_labels:
        sentiment_counts[label] += 1

    # Output sentiment counts to file
    output_file_path = os.path.join('sentiment', f"{model_name}_sentiment_counts.txt")
    with open(output_file_path, "w") as file:
        for label, count in sentiment_counts.items():
            file.write(f"{label} Sentiment Count: {count}\n")

    # Output average sentiment to file
    output_file_path = os.path.join('sentiment', f"{model_name}__average_sentiment.txt")
    with open(output_file_path, "w") as file:
        file.write("Average Sentiment for Liberal Sentences (TextBlob): " + str(average_liberal_sentiment_blob) + "\n")
        file.write("Average Sentiment for Liberal Sentences (VADER): " + str(average_liberal_sentiment_vader) + "\n")
        file.write(
            "Average Sentiment for Conservative Sentences (TextBlob): " + str(average_conservative_sentiment_blob) + "\n")
        file.write(
            "Average Sentiment for Conservative Sentences (VADER): " + str(average_conservative_sentiment_vader) + "\n")

    # Output maximum and minimum sentiment scores to file
    output_file_path = os.path.join('sentiment', f"{model_name}_sentiment_scores.txt")
    with open(output_file_path, "w") as file:
        file.write("Maximum Sentiment for Liberal Sentences (TextBlob): " + str(max_liberal_sentiment_blob) + "\n")
        file.write("Minimum Sentiment for Liberal Sentences (TextBlob): " + str(min_liberal_sentiment_blob) + "\n")
        file.write("Maximum Sentiment for Liberal Sentences (VADER): " + str(max_liberal_sentiment_vader) + "\n")
        file.write("Minimum Sentiment for Liberal Sentences (VADER): " + str(min_liberal_sentiment_vader) + "\n")
        file.write(
            "Maximum Sentiment for Conservative Sentences (TextBlob): " + str(max_conservative_sentiment_blob) + "\n")
        file.write(
            "Minimum Sentiment for Conservative Sentences (TextBlob): " + str(min_conservative_sentiment_blob) + "\n")
        file.write(
            "Maximum Sentiment for Conservative Sentences (VADER): " + str(max_conservative_sentiment_vader) + "\n")
        file.write(
            "Minimum Sentiment for Conservative Sentences (VADER): " + str(min_conservative_sentiment_vader) + "\n")
