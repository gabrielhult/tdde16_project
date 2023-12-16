import add_sentiment as asent
import pandas as pd

def initVader():
    sent_dataset = asent.add_sentiment()
    data = pd.read_csv(sent_dataset)
    print(data.head())



if __name__ == "__main__":
    initVader()