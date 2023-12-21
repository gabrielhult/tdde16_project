import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

def process_text(text):
    return ' '.join([token.lemma_ for token in nlp(text) if token.is_alpha and not token.is_stop])

def pre_process(col_text):
    processed_texts = []

    # Use tqdm to create a progress bar
    for text in tqdm(col_text, desc="Processing text", unit="text"):
        processed_text = process_text(text)
        processed_texts.append(processed_text)

    return processed_texts
