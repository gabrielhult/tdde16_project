import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def multinomial():
    # Load data
    train_data = pd.read_csv('data/x_train_data.csv')
    test_data = pd.read_csv('data/x_test_data.csv')
    train_labels = pd.read_csv('data/y_train_data.csv')
    test_labels = pd.read_csv('data/y_test_data.csv')

    # Load vectorizer
    with open('data/vectoriser.pkl', 'rb') as file:
        vectoriser = pickle.load(file)

    # Vectorise data
    train_data = vectoriser.transform(train_data['Title_Text'])
    test_data = vectoriser.transform(test_data['Title_Text'])

    # Map class labels to numerical values
    class_mapping = {'Conservative': 0, 'Liberal': 1}
    train_labels_mapped = train_labels['Political Lean'].map(class_mapping)
    test_labels_mapped = test_labels['Political Lean'].map(class_mapping)

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
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

    # Visualize results
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    scores = [accuracy, precision, recall, f1]
    plt.bar(metrics, scores)
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Multinomial Naive Bayes Classifier Results')
    plt.show()

    for metric, score in zip(metrics, scores):
        plt.text(metric, score, f'{score:.2f}', ha='center', va='bottom')

    # Save model
    model_file_name = 'multinomial_bayes.sav'
    with open(os.path.join('models', model_file_name), 'wb') as file:
        pickle.dump(model, file)


def bernoulli():
    train_data = pd.read_csv('data/x_train_data.csv')
    test_data = pd.read_csv('data/x_test_data.csv')
    train_labels = pd.read_csv('data/y_train_data.csv')
    test_labels = pd.read_csv('data/y_test_data.csv')

    # Load vectorizer
    with open('data/vectoriser.pkl', 'rb') as file:
        vectoriser = pickle.load(file)

    # Vectorise data
    train_data = vectoriser.transform(train_data['Title_Text'])
    test_data = vectoriser.transform(test_data['Title_Text'])

    # Map class labels to numerical values
    class_mapping = {'Conservative': 0, 'Liberal': 1}
    train_labels_mapped = train_labels['Political Lean'].map(class_mapping)
    test_labels_mapped = test_labels['Political Lean'].map(class_mapping)

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
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

    # Visualize results
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    scores = [accuracy, precision, recall, f1]
    plt.bar(metrics, scores)
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Bernoulli Naive Bayes Classifier Results')
    plt.show()

    for metric, score in zip(metrics, scores):
        plt.text(metric, score, f'{score:.2f}', ha='center', va='bottom')

    # Save model
    model_file_name = 'bernoulli_bayes.sav'
    with open(os.path.join('models', model_file_name), 'wb') as file:
        pickle.dump(model, file)