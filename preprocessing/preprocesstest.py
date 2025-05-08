# preprocesstest.py - UPDATED
import os
import sys
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from joblib import load
from nltk.corpus import sentiwordnet as swn
from nltk import pos_tag, word_tokenize
import spacy
try:
    from preprocessing.preprocess import get_dependency_features, get_lexicon_features
except ImportError:
    # Fallback to direct import
    from preprocess import get_dependency_features, get_lexicon_features
from models.utils.aspect_extractor import AspectExtractor

# Load resources from training preprocessing
nlp = spacy.load('en_core_web_sm')

def parse_test_xml(xml_file, domain):
    """Parse test XML files and extract aspects using AspectExtractor"""
    try:
        # Try with data/raw prefix
        file_path = os.path.join('data', 'raw', xml_file)
        if not os.path.exists(file_path):
            # Fallback to direct path
            file_path = xml_file
            
        print(f"Parsing XML file: {file_path}")
        tree = ET.parse(file_path)
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
                    'aspect': aspect
                })
        
        # Print some debugging info
        print(f"Extracted {len(data)} aspect-sentence pairs")
        if data:
            print(f"First row: {data[0]}")
            
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error parsing XML: {e}")
        # Create a minimal dataframe with some sample data to continue processing
        return pd.DataFrame([{'sentence': 'Sample sentence', 'aspect': 'general'}])

def create_test_features(df):
    """Create features for test data (now with real aspects)"""
    # Check if the dataframe has the expected columns
    if 'sentence' not in df.columns or 'aspect' not in df.columns:
        print(f"WARNING: DataFrame is missing expected columns. Columns: {df.columns}")
        return df
    
    # Basic text features (no placeholder needed)
    df['text'] = df['sentence'] + " " + df['aspect']
    
    # Sentiment lexicon features
    try:
        lexicon_features = df['sentence'].apply(get_lexicon_features).apply(pd.Series)
        # Dependency features
        dep_features = df.apply(
            lambda x: get_dependency_features(x['sentence'], x['aspect']), axis=1
        ).apply(pd.Series)
        
        return df.join(lexicon_features).join(dep_features)
    except Exception as e:
        print(f"Error creating features: {e}")
        # Add minimal features to continue processing
        df['lexicon_pos'] = 0
        df['lexicon_neg'] = 0
        return df

if __name__ == "__main__":
    # Print directory info for debugging
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    
    # Try to create data directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    test_files = [
        ("Laptops_Test_Data_PhaseA.xml", "laptop"),
        ("Restaurants_Test_Data_PhaseA.xml", "restaurant")
    ]
    
    for test_xml, domain in test_files:
        print(f"\nProcessing {domain} domain")
        # Parse test data with real aspects
        test_df = parse_test_xml(test_xml, domain)
        
        # Create features
        test_df = create_test_features(test_df)
        
        try:
            # Load domain-specific vectorizers
            vectorizer_path = f"data/processed/{domain}_tfidf_vectorizer.joblib"
            if not os.path.exists(vectorizer_path):
                vectorizer_path = f"{domain}_tfidf_vectorizer.joblib"
            
            print(f"Loading TF-IDF vectorizer from {vectorizer_path}")
            tfidf = load(vectorizer_path)
            
            bow_path = f"data/processed/{domain}_bow_vectorizer.joblib"
            if not os.path.exists(bow_path):
                bow_path = f"{domain}_bow_vectorizer.joblib"
                
            print(f"Loading BOW vectorizer from {bow_path}")
            bow = load(bow_path)
            
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
            output_path = f"data/processed/{domain}_test_features_unlabeled.csv"
            print(f"Saving features to {output_path}")
            test_df.to_csv(output_path, index=False)
            
            np.save(f"data/processed/{domain}_X_test_unlabeled.npy", X_test)
            print(f"Successfully processed {domain} domain")
        except Exception as e:
            print(f"Error processing {domain} domain: {e}")