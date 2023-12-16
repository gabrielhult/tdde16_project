import add_sentiment as asent

def initVader():
    sent_dataset = asent.add_sentiment()
    print(sent_dataset)



if __name__ == "__main__":
    initVader()