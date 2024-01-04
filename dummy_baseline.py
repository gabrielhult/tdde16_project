import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import data_mapping as dm
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def dummy(vectoriser):
    train_labels_mapped, test_labels_mapped, train_data, test_data = dm()

    # Train model
    model = DummyClassifier(strategy='most_frequent').fit(train_data, train_labels_mapped.values.ravel())

    # Predict
    predicted = model.predict(test_data)

    # Evaluate
    accuracy = accuracy_score(test_labels_mapped, predicted)
    precision = precision_score(test_labels_mapped, predicted, average='binary')
    recall = recall_score(test_labels_mapped, predicted)
    f1 = f1_score(test_labels_mapped, predicted)
    print(f"Dummy Classifier:")
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
    plt.title('Confusion Matrix - Dummy Classifier')
    plt.savefig('/home/gult/tdde16_project/graphs/dummy_confusion_matrix.pdf')

    # Save model
    model_file_name = 'dummy_classifier.sav'
    with open(os.path.join('models', model_file_name), 'wb') as file:
        pickle.dump(model, file)
