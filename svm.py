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


def linear(vectoriser):
    train_labels_mapped, test_labels_mapped, train_data, test_data = dm()

    leans = sorted(train_labels_mapped.unique())

    # Train model
    model = LinearSVC(dual=True).fit(train_data, train_labels_mapped)

    # Print out the most decisive words for classifying as either "Liberal" or "Conservative"
    decisive_words(model, vectoriser, 'linear')

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
    
def decisive_words(model, vectoriser, model_name):
    coefficients = model.coef_

    # Get the feature names (words)
    feature_names = np.array(vectoriser.get_feature_names_out())

    # Assuming you want the top N words to display
    top_n_words = 10

    # Display top words for each class
    for class_index in range(coefficients.shape[0]):
        
        # Sort the features based on the coefficients for the current class
        sorted_indices_svc = np.argsort(coefficients[class_index])[::-1]

        # Get the top N words and their corresponding coefficients for the current class
        top_words_with_coefficients_svc = list(zip(feature_names[sorted_indices_svc][:top_n_words],
                                                coefficients[class_index][sorted_indices_svc][:top_n_words]))

        # Save the top words with their coefficients differences to a text file
        label_name = "liberal" if class_index == 0 else "conservative"
        output_file_path = os.path.join('top_words', f"{model_name}_{label_name}_words.txt")
        with open(output_file_path, "w") as file:
            file.write(f"Top {top_n_words} words for classifying as '{label_name}':\n")
            for word, coefficient_value in top_words_with_coefficients_svc:
                file.write(f"Word: {word}, Coefficient: {coefficient_value}\n")
                
