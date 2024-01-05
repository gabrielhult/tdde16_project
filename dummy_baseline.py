import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import data_mapping as dm
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def dummy(vectoriser):
    train_labels_mapped, test_labels_mapped, train_data, test_data = dm()

    leans = sorted(train_labels_mapped.unique())

    # Train model
    model = DummyClassifier(strategy='uniform').fit(train_data, train_labels_mapped)

    # Predict
    predicted = model.predict(test_data)

    # Evaluate
    report = classification_report(test_labels_mapped, predicted, target_names=leans, zero_division=np.nan)
    print(f"Dummy Classifier:\n", report)

    # Plot confusion matrix
    cm = confusion_matrix(test_labels_mapped, predicted)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Dummy Classifier')
    plt.savefig('/home/gult/tdde16_project/graphs/dummy_confusion_matrix.pdf')

    # Save model
    model_file_name = 'dummy_classifier.sav'
    with open(os.path.join('models', model_file_name), 'wb') as file:
        pickle.dump(model, file)
