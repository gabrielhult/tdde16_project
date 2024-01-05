import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data_utils import data_mapping as dm
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def multinomial(vectoriser):
    train_labels_mapped, test_labels_mapped, train_data, test_data = dm()

    leans = sorted(train_labels_mapped.unique())
    # Train model
    model = MultinomialNB().fit(train_data, train_labels_mapped)

    # Print out the most decisive words for classifying as either "Liberal" or "Conservative"
    decisive_words(model, vectoriser, 'multinomial')

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
    train_labels_mapped, test_labels_mapped, train_data, test_data = dm()

    leans = sorted(train_labels_mapped.unique())

    # Train model
    model = BernoulliNB().fit(train_data, train_labels_mapped)

    # Print out the most decisive words for classifying as either "Liberal" or "Conservative"
    decisive_words(model, vectoriser, 'bernoulli')

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
    # Get the feature log probabilities for each class
    feature_log_probs = model.feature_log_prob_

    # Get the feature names (words)
    feature_names = np.array(vectoriser.get_feature_names_out())

    # Assuming you want to compare the classes 0 (liberal) and 1 (conservative)
    liberal_class_index = 0
    conservative_class_index = 1
    top_n_words = 10  # Choose the number of top words to display

    # Calculate the log probability differences separately for each class
    log_prob_diff_liberal = feature_log_probs[liberal_class_index] - feature_log_probs[conservative_class_index]
    log_prob_diff_conservative = -log_prob_diff_liberal  # Difference for conservative is negative of liberal

    # Sort the features based on the log probability differences for liberal
    sorted_indices_liberal = np.argsort(log_prob_diff_liberal)[::-1]
    
    # Get the top N words and their corresponding log probability differences for liberal
    top_words_with_log_prob_diff_liberal = list(zip(feature_names[sorted_indices_liberal][:top_n_words],
                                                    log_prob_diff_liberal[sorted_indices_liberal][:top_n_words]))

    # Sort the features based on the log probability differences for conservative
    sorted_indices_conservative = np.argsort(log_prob_diff_conservative)[::-1]

    # Get the top N words and their corresponding log probability differences for conservative
    top_words_with_log_prob_diff_conservative = list(zip(feature_names[sorted_indices_conservative][:top_n_words],
                                                    log_prob_diff_conservative[sorted_indices_conservative][:top_n_words]))

    # Save the top words with their log probability differences for liberal to a text file
    output_file_path = os.path.join('top_words', f"{model_name}_liberal_words.txt")
    with open(output_file_path, "w") as file:
        file.write(f"Top {top_n_words} words for classifying as 'liberal':\n")
        for word, log_prob_diff_value in top_words_with_log_prob_diff_liberal:
            file.write(f"Word: {word}, Log Probability Difference: {log_prob_diff_value}\n")

    # Save the top words with their log probability differences for conservative to a text file
    output_file_path = os.path.join('top_words', f"{model_name}_conservative_words.txt")
    with open(output_file_path, "w") as file:
        file.write(f"Top {top_n_words} words for classifying as 'conservative':\n")
        for word, log_prob_diff_value in top_words_with_log_prob_diff_conservative:
            file.write(f"Word: {word}, Log Probability Difference: {log_prob_diff_value}\n")
