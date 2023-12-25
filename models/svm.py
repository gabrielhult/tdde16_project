from sklearn.svm import LinearSVC
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pickle




def linear():
    # Load data
    train_data = pd.read_csv('data/x_train_data.csv')
    test_data = pd.read_csv('data/x_test_data.csv')
    train_labels = pd.read_csv('data/y_train_data.csv')
    test_labels = pd.read_csv('data/y_test_data.csv')
    print(train_data.head())
    print(test_data.head())
    print(train_labels.head())
    print(test_labels.head())

    # Train model
    model = LinearSVC().fit(train_data, train_labels.values.ravel())

    # Predict
    predicted = model.predict(test_data)
    print(predicted)

    # Evaluate
    print(f"Linear SVM Classifier:")
    print(f"Accuracy: {accuracy_score(test_labels, predicted)}")
    print(f"Precision: {precision_score(test_labels, predicted)}")
    print(f"Recall: {recall_score(test_labels, predicted)}")
    print(f"F1: {f1_score(test_labels, predicted)}")

    # Save model
    model_file_name = 'linear_svm.sav'
    with open(os.path.join('models', model_file_name), 'wb') as file:
        pickle.dump(model, file)
    

def rbf():
    # Load data
    train_data = pd.read_csv('data/x_train_data.csv')
    test_data = pd.read_csv('data/x_test_data.csv')
    train_labels = pd.read_csv('data/y_train_data.csv')
    test_labels = pd.read_csv('data/y_test_data.csv')
    print(train_data.head())
    print(test_data.head())
    print(train_labels.head())
    print(test_labels.head())

    # Train model
    model = LinearSVC().fit(train_data, train_labels.values.ravel())

    # Predict
    predicted = model.predict(test_data)
    print(predicted)

    # Evaluate
    print(f"RBF SVM Classifier:")
    print(f"Accuracy: {accuracy_score(test_labels, predicted)}")
    print(f"Precision: {precision_score(test_labels, predicted)}")
    print(f"Recall: {recall_score(test_labels, predicted)}")
    print(f"F1: {f1_score(test_labels, predicted)}")

    # Save model
    model_file_name = 'rbf_svm.sav'
    with open(os.path.join('models', model_file_name), 'wb') as file:
        pickle.dump(model, file)