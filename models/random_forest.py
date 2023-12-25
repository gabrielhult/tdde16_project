from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pickle
import matplotlib.pyplot as plt
from data_utils import data_mapping as dm


def forest():
    train_labels_mapped, test_labels_mapped, train_data, test_data = dm()

    # Train model
    model = RandomForestClassifier().fit(train_data, train_labels_mapped.values.ravel())

    # Predict
    predicted = model.predict(test_data)
    print(predicted)

    # Evaluate
    accuracy = accuracy_score(test_labels_mapped, predicted)
    precision = precision_score(test_labels_mapped, predicted, average='binary')
    recall = recall_score(test_labels_mapped, predicted)
    f1 = f1_score(test_labels_mapped, predicted)
    print(f"Accuracy: {round(accuracy, 3)}")
    print(f"Precision: {round(precision, 3)}")
    print(f"Recall: {round(recall, 3)}")
    print(f"F1: {round(f1, 3)}")

    # Visualize results
    # metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    # scores = [accuracy, precision, recall, f1]
    # plt.bar(metrics, scores)
    # plt.xlabel('Metrics')
    # plt.ylabel('Scores')
    # plt.title('Random Forest Classifier Results')
    plt.show()

    # for metric, score in zip(metrics, scores):
    #     plt.text(metric, score, f'{score:.2f}', ha='center', va='bottom')

    # Save model
    model_file_name = 'random_forest.sav'
    with open(os.path.join('models', model_file_name), 'wb') as file:
        pickle.dump(model, file)
    

