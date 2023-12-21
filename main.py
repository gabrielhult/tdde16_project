#import add_sentiment as asent
import split_train_test as stt
import pandas as pd
import os

# def initVader():
#     sent_dataset = asent.add_sentiment()
#     data = pd.read_csv(sent_dataset)
#     print(data.head())

def split_dataset():
    train_csv, test_csv = stt.split_data()
    train_csv_path = os.path.join("data", train_csv)
    test_csv_path = os.path.join("data", test_csv)
    print("gsmkgsgm")
    train_data = pd.read_csv(train_csv_path)
    test_data = pd.read_csv(test_csv_path)
    print(train_data.head())
    print(test_data.head())



if __name__ == "__main__":
    split_dataset()
    #initVader()