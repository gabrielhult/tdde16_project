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


def forest():
    train_labels_mapped, test_labels_mapped, train_data, test_data, data = dm()

    leans = sorted(train_labels_mapped.unique())

    # Train model
    model = RandomForestClassifier().fit(train_data, train_labels_mapped.values.ravel())

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