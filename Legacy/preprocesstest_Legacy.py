# File: preprocess_test.py
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from joblib import load
from nltk.corpus import sentiwordnet as swn
from nltk import pos_tag, word_tokenize
import spacy
from preprocess import get_dependency_features,get_lexicon_features
from aspect_extractor import AspectExtractor

# Load resources from training preprocessing
nlp = spacy.load('en_core_web_sm')
# tfidf = load('tfidf_vectorizer.joblib')
# bow = load('bow_vectorizer.joblib')

def parse_test_xml(xml_file, domain):
    """Parse test XML files and extract aspects using AspectExtractor"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = []
    aspect_extractor = AspectExtractor()  # Initialize extractor
    
    for sentence in root.findall('sentence'):
        text = sentence.find('text').text
        text = text.replace('&quot;', '"').replace('&apos;', "'").strip()
        
        # Extract aspects for this sentence
        aspects = aspect_extractor.extract_aspects(text, domain)
        
        # Create a row for each (sentence, aspect) pair
        for aspect in aspects:
            data.append({
                'sentence': text,
                'aspect': aspect  # Use extracted aspect instead of 'general'
            })
    
    return pd.DataFrame(data)

def create_test_features(df):
    """Create features for test data (now with real aspects)"""
    # Basic text features (no placeholder needed)
    df['text'] = df['sentence'] + " " + df['aspect']
    
    # Sentiment lexicon features
    lexicon_features = df['sentence'].apply(get_lexicon_features).apply(pd.Series)
    
    # Dependency features
    dep_features = df.apply(
        lambda x: get_dependency_features(x['sentence'], x['aspect']), axis=1
    ).apply(pd.Series)
    
    return df.join(lexicon_features).join(dep_features)

if __name__ == "__main__":
    test_files = [
        ("Laptops_Test_Data_PhaseA.xml", "laptop"),
        ("Restaurants_Test_Data_PhaseA.xml", "restaurant")
    ]
    
    for test_xml, domain in test_files:
        # Parse test data with real aspects
        test_df = parse_test_xml(test_xml, domain)
        
        # Create features
        test_df = create_test_features(test_df)
        
        # Load domain-specific vectorizers
        tfidf = load(f"{domain}_tfidf_vectorizer.joblib")
        bow = load(f"{domain}_bow_vectorizer.joblib")
        
        # Transform features
        X_test_tfidf = tfidf.transform(test_df['text'])
        X_test_bow = bow.transform(test_df['text'])
        
        # Combine features
        X_test = np.hstack([
            X_test_tfidf.toarray(),
            X_test_bow.toarray(),
            test_df[['lexicon_pos', 'lexicon_neg']].values
        ])
        
        # Save test data with extracted aspects
        test_df.to_csv(f"{domain}_test_features_unlabeled.csv", index=False)
        np.save(f"{domain}_X_test_unlabeled.npy", X_test)
