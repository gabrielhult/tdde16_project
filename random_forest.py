import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data_utils import data_mapping as dm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def forest(vectoriser):
    train_labels_mapped, test_labels_mapped, train_data, test_data, data = dm()

    leans = sorted(train_labels_mapped.unique())

    # Train model
    model = RandomForestClassifier().fit(train_data, train_labels_mapped.values.ravel())

    # Print out the most decisive words for classifying as either "Liberal" or "Conservative"
    # liberal_top_words, conservative_top_words = decisive_words(model, vectoriser, 'random_forest')

    # sentiment_prediction(data, liberal_top_words, conservative_top_words, 'random_forest')

    # Predict
    predicted = model.predict(test_data)

    # Evaluate
    report = classification_report(test_labels_mapped, predicted, target_names=leans, zero_division=np.nan)
    print(f"Random Forest Classifier:\n", report)
    

    # Plot confusion matrix
    cm = confusion_matrix(test_labels_mapped, predicted)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Random Forest Classifier')
    plt.savefig('/home/gult/tdde16_project/graphs/forest_confusion_matrix.pdf')

    # Save model
    model_file_name = 'random_forest.sav'
    with open(os.path.join('models', model_file_name), 'wb') as file:
        pickle.dump(model, file)

def decisive_words(data, model, vectoriser, model_name):
    # Get feature log probabilities
    feature_log_probs = model.feature_log_prob_

    # Top words for each class
    liberal_top_words = [vectoriser.get_feature_names_out()[i] for i in feature_log_probs[0].argsort()[-10:][::-1]]
    conservative_top_words = [vectoriser.get_feature_names_out()[i] for i in feature_log_probs[1].argsort()[-10:][::-1]]

    # Print or visualize the top words for each class
    print("Top words for Liberal:", liberal_top_words)
    print("Top words for Conservative:", conservative_top_words)

    # Save the top words with their log probability differences for liberal to a text file
    output_file_path = os.path.join('top_words', f"{model_name}_liberal_words.txt")
    with open(output_file_path, "w") as file:
        file.write(f"Top 10 words for classifying as 'liberal':\n")
        for word in liberal_top_words:
            file.write(f"Word: {word} \n")
            vader_positive_percentage, vader_neutral_percentage, vader_negative_percentage, blob_positive_percentage, blob_neutral_percentage, blob_negative_percentage = sentiment_prediction(data, word)
            file.write(f"Sentiment scores (VADER): \n\tPositive={vader_positive_percentage:.2f}%, \n\tNeutral={vader_neutral_percentage:.2f}%, \n\tNegative={vader_negative_percentage:.2f}%\n")
            file.write(f"Sentiment scores (TextBlob): \n\tPositive={blob_positive_percentage:.2f}%, \n\tNeutral={blob_neutral_percentage:.2f}%, \n\tNegative={blob_negative_percentage:.2f}%\n")

    # Save the top words with their log probability differences for conservative to a text file
    output_file_path = os.path.join('top_words', f"{model_name}_conservative_words.txt")
    with open(output_file_path, "w") as file:
        file.write(f"Top 10 words for classifying as 'conservative':\n")
        for word in conservative_top_words:
            file.write(f"Word: {word} \n")
            vader_positive_percentage, vader_neutral_percentage, vader_negative_percentage, blob_positive_percentage, blob_neutral_percentage, blob_negative_percentage = sentiment_prediction(data, word)
            file.write(f"Sentiment scores (VADER): \n\tPositive={vader_positive_percentage:.2f}%, \n\tNeutral={vader_neutral_percentage:.2f}%, \n\tNegative={vader_negative_percentage:.2f}%\n")
            file.write(f"Sentiment scores (TextBlob): \n\tPositive={blob_positive_percentage:.2f}%, \n\tNeutral={blob_neutral_percentage:.2f}%, \n\tNegative={blob_negative_percentage:.2f}%\n")

    return liberal_top_words, conservative_top_words


def sentiment_prediction(data, word):
    # Identify rows with the specific word
    rows_with_word = data[data['Title_Text'].str.contains(word, case=False)]

    # Extract sentences with the specific word
    sentences = rows_with_word['Title_Text'].tolist()

    # Perform sentiment analysis using TextBlob
    def get_sentiment_textblob(sentence):
        analysis = TextBlob(sentence)
        return analysis.sentiment.polarity

    # Perform sentiment analysis using VADER
    def get_sentiment_vader(sentence):
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = analyzer.polarity_scores(sentence)
        return sentiment_scores['compound']

    # Analyze sentiment for the sentences
    sentiments_blob = [get_sentiment_textblob(sentence) for sentence in sentences]
    sentiments_vader = [get_sentiment_vader(sentence) for sentence in sentences]

    # Calculate the percentage of positive, neutral, and negative sentiments for VADER
    vader_positive_percentage = sum(sentiment > 0 for sentiment in sentiments_vader) / len(sentiments_vader) * 100
    vader_neutral_percentage = sum(sentiment == 0 for sentiment in sentiments_vader) / len(sentiments_vader) * 100
    vader_negative_percentage = sum(sentiment < 0 for sentiment in sentiments_vader) / len(sentiments_vader) * 100

    # Calculate the percentage of positive, neutral, and negative sentiments for TextBlob
    blob_positive_percentage = sum(sentiment > 0 for sentiment in sentiments_blob) / len(sentiments_blob) * 100
    blob_neutral_percentage = sum(sentiment == 0 for sentiment in sentiments_blob) / len(sentiments_blob) * 100
    blob_negative_percentage = sum(sentiment < 0 for sentiment in sentiments_blob) / len(sentiments_blob) * 100

    return vader_positive_percentage, vader_neutral_percentage, vader_negative_percentage, blob_positive_percentage, blob_neutral_percentage, blob_negative_percentage
