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

def linear():
    train_labels_mapped, test_labels_mapped, train_data, test_data, data = dm()

    # Check if there are only two unique classes
    unique_classes = train_labels_mapped.nunique()
    if unique_classes != 2:
        raise ValueError(f"Linear SVM requires a binary classification problem, but found {unique_classes} classes.")

    leans = sorted(train_labels_mapped.unique())

    # Train model
    model = LinearSVC(dual='auto').fit(train_data, train_labels_mapped)

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
