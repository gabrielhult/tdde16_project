import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from data_utils import data_mapping as dm
import seaborn as sns

def multinomial():
    train_labels_mapped, test_labels_mapped, train_data, test_data = dm()

    # Train model
    model = MultinomialNB().fit(train_data, train_labels_mapped.values.ravel())

    # Predict
    predicted = model.predict(test_data)
    print(predicted)

    # Evaluate
    accuracy = accuracy_score(test_labels_mapped, predicted)
    precision = precision_score(test_labels_mapped, predicted, average='binary')
    recall = recall_score(test_labels_mapped, predicted)
    f1 = f1_score(test_labels_mapped, predicted)
    print(f"Multinomial Naive Bayes Classifier:")
    print(f"Accuracy: {round(accuracy, 3)}")
    print(f"Precision: {round(precision, 3)}")
    print(f"Recall: {round(recall, 3)}")
    print(f"F1: {round(f1, 3)}")

    # Plot confusion matrix
    cm = confusion_matrix(test_labels_mapped, predicted)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Multinomial Naive Bayes Classifier')
    plt.show()

    # Save model
    model_file_name = 'multinomial_bayes.sav'
    with open(os.path.join('models', model_file_name), 'wb') as file:
        pickle.dump(model, file)


def bernoulli():
    train_labels_mapped, test_labels_mapped, train_data, test_data = dm()

    # Train model
    model = BernoulliNB().fit(train_data, train_labels_mapped.values.ravel())

    # Predict
    predicted = model.predict(test_data)
    print(predicted)

    # Evaluate
    accuracy = accuracy_score(test_labels_mapped, predicted)
    precision = precision_score(test_labels_mapped, predicted, average='binary')
    recall = recall_score(test_labels_mapped, predicted)
    f1 = f1_score(test_labels_mapped, predicted)
    print(f"Bernoulli Naive Bayes Classifier:")
    print(f"Accuracy: {round(accuracy, 3)}")
    print(f"Precision: {round(precision, 3)}")
    print(f"Recall: {round(recall, 3)}")
    print(f"F1: {round(f1, 3)}")

    # Plot confusion matrix
    cm = confusion_matrix(test_labels_mapped, predicted)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Bernoulli Naive Bayes Classifier')
    plt.show()

    # Save model
    model_file_name = 'bernoulli_bayes.sav'
    with open(os.path.join('models', model_file_name), 'wb') as file:
        pickle.dump(model, file)