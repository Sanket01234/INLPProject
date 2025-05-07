import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os
import json
from nltk.corpus import sentiwordnet as swn
from nltk import pos_tag, word_tokenize
import spacy
import nltk
from tqdm import tqdm

# Make sure necessary NLTK resources are downloaded
try:
    nltk.download('sentiwordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    print("Could not download NLTK resources. If needed, please install manually.")

# Initialize spaCy
try:
    nlp = spacy.load('en_core_web_sm')
except:
    print("Could not load spaCy model. Please install with: python -m spacy download en_core_web_sm")

def parse_xml(xml_file, domain):
    """Parse XML files and extract aspect-sentence pairs (aspect terms only)"""
    print(f"Parsing {xml_file} for {domain} domain...")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = []
    
    for sentence in root.findall('sentence'):
        text = sentence.find('text').text
        text = text.replace('&quot;', '"').replace('&apos;', "'").strip()
        aspects = []

        for term in sentence.findall('aspectTerms/aspectTerm'):
            aspect = term.get('term').replace('&quot;', '"').strip()
            polarity = term.get('polarity')
            aspects.append((aspect, polarity))

        for aspect, polarity in aspects:
            data.append({
                'sentence': text,
                'aspect': aspect,
                'polarity': polarity.lower() if polarity else 'neutral'
            })
    
    return pd.DataFrame(data)

def extract_aspect_terms(df, domain):
    """Extract and save unique aspect terms per domain"""
    aspects = set(df['aspect'].str.lower())
    os.makedirs('domain_data', exist_ok=True)
    with open(f'domain_data/{domain}_aspects.json', 'w') as f:
        json.dump(list(aspects), f)
    print(f"Extracted {len(aspects)} unique aspects for {domain} domain")
    return aspects

# def get_lexicon_features(text):
#     """Extract sentiment lexicon scores using SentiWordNet"""
#     pos_tags = pos_tag(word_tokenize(text))
#     pos_scores = []
#     neg_scores = []
    
#     for word, tag in pos_tags:
#         synsets = list(swn.senti_synsets(word))
#         if synsets:
#             pos_scores.append(synsets[0].pos_score())
#             neg_scores.append(synsets[0].neg_score())
    
#     return {
#         'lexicon_pos': np.mean(pos_scores) if pos_scores else 0,
#         'lexicon_neg': np.mean(neg_scores) if neg_scores else 0
#     }

def preprocess_for_bert(xml_file, domain, output_dir='bert_data'):
    """Preprocess data for BERT model"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse XML
    df = parse_xml(xml_file, domain)
    
    # Filter valid polarities
    df = df[df['polarity'].isin(['positive', 'negative', 'neutral'])]
    
    # Extract aspects
    extract_aspect_terms(df, domain)
    
    # # Add lexicon features (can be useful for hybrid models later)
    # print(f"Extracting lexicon features for {domain}...")
    # lexicon_features = []
    # for text in tqdm(df['sentence']):
    #     lexicon_features.append(get_lexicon_features(text))
    
    # lexicon_df = pd.DataFrame(lexicon_features)
    # df = pd.concat([df, lexicon_df], axis=1)
    
    # Split data at sentence level to maintain coherence
    sentences = df.groupby('sentence').agg(list).reset_index()
    
    # Get class counts for stratification
    sentences['polarity_tuple'] = sentences['polarity'].apply(tuple)
    class_counts = sentences['polarity_tuple'].value_counts()
    valid_classes = class_counts[class_counts > 1].index
    
    # Filter to valid classes (needed for stratification)
    sentences = sentences[sentences['polarity_tuple'].isin(valid_classes)]
    
    # Split data (80% train, 20% validation)
    from sklearn.model_selection import train_test_split
    train_sents, val_sents = train_test_split(
        sentences, 
        test_size=0.2, 
        stratify=sentences['polarity_tuple'],
        random_state=42
    )
    
    # Convert back to individual aspect-sentence pairs
    train_df = train_sents.explode(['aspect', 'polarity']).reset_index(drop=True)
    val_df = val_sents.explode(['aspect', 'polarity']).reset_index(drop=True)
    
    # Save to CSV
    train_df.to_csv(f"{output_dir}/{domain}_train_features.csv", index=False)
    val_df.to_csv(f"{output_dir}/{domain}_val_features.csv", index=False)
    
    print(f"Saved {len(train_df)} training and {len(val_df)} validation examples for {domain}")
    
    return train_df, val_df

def prepare_test_data(xml_file, domain, output_dir='bert_data'):
    """Prepare test data with extracted aspects using AspectExtractor"""
    from aspect_extractor import AspectExtractor  # Local import to avoid circular dependencies
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Parse XML and extract sentences
    print(f"Parsing {xml_file} for {domain} test data...")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = []
    
    aspect_extractor = AspectExtractor()
    
    for sentence in root.findall('sentence'):
        text = sentence.find('text').text
        text = text.replace('&quot;', '"').replace('&apos;', "'").strip()
        
        # Extract aspects using AspectExtractor
        aspects = aspect_extractor.extract_aspects(text, domain)
        
        # Create a row for each (sentence, aspect) pair
        for aspect in aspects:
            data.append({
                'sentence': text,
                'aspect': aspect
            })
    
    # Create DataFrame
    test_df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = os.path.join(output_dir, f"{domain}_test_features.csv")
    test_df.to_csv(output_path, index=False)
    print(f"Saved {len(test_df)} test examples for {domain} to {output_path}")
    
    return test_df

if __name__ == "__main__":
    data_files = [
        ("./dataset/Laptop_Train_v2.xml", "laptop", "train"),
        ("./dataset/Restaurants_Train_v2.xml", "restaurant", "train"),
        ("./dataset/Laptops_Test_Data_PhaseA.xml", "laptop", "test"),
        ("./dataset/Restaurants_Test_Data_PhaseA.xml", "restaurant", "test")
    ]
    
    output_dir = 'bert_data'
    
    for xml_file, domain, split in data_files:
        print(f"Processing {domain} {split} data...")
        if split == 'train':
            preprocess_for_bert(xml_file, domain, output_dir)
        else:
            prepare_test_data(xml_file, domain, output_dir)
    
    print("Preprocessing complete!")