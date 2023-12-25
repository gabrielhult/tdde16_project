import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data_utils import data_mapping as dm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def forest(vectoriser):
    train_labels_mapped, test_labels_mapped, train_data, test_data = dm()

    # Train model
    model = RandomForestClassifier().fit(train_data, train_labels_mapped.values.ravel())

    # Print out the most decisive words for classifying as either "Liberal" or "Conservative"
    decisive_words(model, vectoriser, 'random_forest')

    # Predict
    predicted = model.predict(test_data)

    # Evaluate
    accuracy = accuracy_score(test_labels_mapped, predicted)
    precision = precision_score(test_labels_mapped, predicted, average='binary')
    recall = recall_score(test_labels_mapped, predicted)
    f1 = f1_score(test_labels_mapped, predicted)
    print(f"Random Forest Classifier:")
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
    plt.title('Confusion Matrix - Random Forest Classifier')
    plt.savefig('/home/gult/tdde16_project/graphs/forest_confusion_matrix.png')

    # Save model
    model_file_name = 'random_forest.sav'
    with open(os.path.join('models', model_file_name), 'wb') as file:
        pickle.dump(model, file)
    
def decisive_words(model, vectoriser, model_name):
    # Get feature importances from the trained model for both classes
    importances = model.feature_importances_

    # Get the feature names (words)
    feature_names = np.array(vectoriser.get_feature_names_out())

    # Assuming you want the top N words to display
    top_n_words = 10

    # Display top words for each class
    for class_index in [0, 1]:
        # Sort the features based on the importances for the current class
        sorted_indices_rf = np.argsort(importances[:, class_index])[::-1]

        # Get the top N words and their corresponding importances for the current class
        top_words_with_importances_rf = list(zip(feature_names[sorted_indices_rf][:top_n_words],
                                                importances[sorted_indices_rf][:top_n_words, class_index]))

        # Print or analyze the top words with their importances for the current class
        label_name = "liberal" if class_index == 0 else "conservative"
        print(f"Top {top_n_words} words for RandomForestClassifier predicting '{label_name}':")
        for word, importance_value in top_words_with_importances_rf:
            print(f"Word: {word}, Importance: {importance_value}")

        # Save the top words with their log probability differences to a text file
        with open(f"{model_name}_{label_name}_words.txt", "w") as file:
            file.write(f"Top {top_n_words} words for classifying as '{label_name}':\n")
            for word, importance_value in top_words_with_importances_rf:
                file.write(f"Word: {word}, Importance: {importance_value}\n")

