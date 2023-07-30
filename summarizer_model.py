import spacy
import numpy as np

def summarize(text, ratio=0.4):
    nlp = spacy.load('en_core_web_lg')
    doc = nlp(text)

    sentences = [sent.text.strip() for sent in doc.sents]
    
    num_sentences = int(len(sentences) * ratio)
    
    distance_matrix = np.zeros((len(sentences), len(sentences)))
    for i, sent1 in enumerate(sentences):
        for j, sent2 in enumerate(sentences):
            if i == j:
                pass
            distance_matrix[i, j] = nlp(sent1).similarity(nlp(sent2))
    
    scores = distance_matrix.sum(axis=0)

    sorted_indices = np.argsort(scores)[::-1]
    return ' '.join([sentences[i] for i in sorted_indices[:num_sentences]])