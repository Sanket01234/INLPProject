import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tabulate import tabulate
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def load_model_predictions(model_type, domain):
    """Load model predictions"""
    try:
        if model_type == 'svm':
            return pd.read_csv(f"outputs/traditional/{domain}_svm_test_predictions.csv")
        elif model_type == 'lr':  # Added Logistic Regression model
            return pd.read_csv(f"outputs/traditional/{domain}_lr_test_predictions.csv")
        elif model_type == 'bert':
            return pd.read_csv(f"outputs/bert/{domain}/{domain}_bert_predictions.csv")
        elif model_type == 'hybrid':
            return pd.read_csv(f"outputs/hybrid/{domain}/predictions.csv")
        return None
    except FileNotFoundError:
        print(f"Predictions file for {model_type} model on {domain} domain not found")
        return None

def analyze_errors_by_sentiment(domain, model):
    """Analyze which sentiments are most challenging"""
    df = load_model_predictions(model, domain)
    if df is None:
        return None
    
    # We can't do error analysis without true labels
    # But we can analyze prediction distribution
    sentiment_counts = df['predicted_polarity'].value_counts()
    print(f"\n{model.upper()} - {domain.upper()} - Prediction Distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count} ({count/len(df)*100:.1f}%)")
    
    return df

def analyze_common_errors(domain, models):
    """Find instances that all models get wrong"""
    # This would require true labels for comparison
    # Instead, we'll analyze disagreements between models
    predictions = {}
    for model in models:
        pred_df = load_model_predictions(model, domain)
        if pred_df is not None:
            # Create a unique identifier for each test example
            pred_df['id'] = pred_df['sentence'] + ' | ' + pred_df['aspect']
            predictions[model] = pred_df.set_index('id')[['sentence', 'aspect', 'predicted_polarity']]
    
    if len(predictions) < 2:
        print("Need at least two models for comparison")
        return
    
    # Get common ids across all models
    common_ids = set.intersection(*[set(preds.index) for preds in predictions.values()])
    
    # Find examples where models disagree
    disagreements = []
    for id in common_ids:
        preds = {m: predictions[m].loc[id, 'predicted_polarity'] for m in models if m in predictions}
        if len(set(preds.values())) > 1:  # Models disagree
            sentence = predictions[models[0]].loc[id, 'sentence']
            aspect = predictions[models[0]].loc[id, 'aspect']
            disagreements.append({
                'sentence': sentence,
                'aspect': aspect,
                **preds
            })
    
    return pd.DataFrame(disagreements)

def analyze_aspects(domain, model):
    """Analyze which aspects are most challenging"""
    df = load_model_predictions(model, domain)
    if df is None:
        return
    
    # Get sentiment distribution by aspect
    aspect_sentiment = df.groupby('aspect')['predicted_polarity'].value_counts().unstack(fill_value=0)
    
    # Calculate total count and dominant sentiment for each aspect
    aspect_sentiment['total'] = aspect_sentiment.sum(axis=1)
    aspect_sentiment['dominant'] = aspect_sentiment.idxmax(axis=1)
    
    # Get top aspects by frequency
    top_aspects = aspect_sentiment.sort_values('total', ascending=False).head(10)
    
    print(f"\n{model.upper()} - {domain.upper()} - Top 10 Aspects:")
    for aspect, row in top_aspects.iterrows():
        print(f"  {aspect}: {row['total']} instances, dominant sentiment: {row['dominant']}")
    
    return aspect_sentiment

def analyze_sentence_length(domain, model):
    """Analyze if sentence length affects predictions"""
    df = load_model_predictions(model, domain)
    if df is None:
        return
    
    # Calculate sentence lengths
    df['sentence_length'] = df['sentence'].apply(len)
    
    # Group by sentiment and get average length
    length_by_sentiment = df.groupby('predicted_polarity')['sentence_length'].agg(['mean', 'median', 'std']).round(1)
    
    print(f"\n{model.upper()} - {domain.upper()} - Sentence Length by Sentiment:")
    print(tabulate(length_by_sentiment, headers=['Mean', 'Median', 'Std Dev'], tablefmt="grid"))
    
    # Create length distribution plot
    plt.figure(figsize=(10, 6))
    for sentiment in df['predicted_polarity'].unique():
        subset = df[df['predicted_polarity'] == sentiment]
        plt.hist(subset['sentence_length'], alpha=0.5, bins=20, label=sentiment)
    
    plt.xlabel('Sentence Length (characters)')
    plt.ylabel('Frequency')
    plt.title(f'Sentence Length Distribution by Sentiment - {model.upper()} - {domain.upper()}')
    plt.legend()
    
    # Save chart
    os.makedirs('outputs/analysis', exist_ok=True)
    plt.savefig(f'outputs/analysis/{domain}_{model}_sentence_length.png')
    
    return length_by_sentiment

def analyze_keywords(domain, model):
    """Find keywords that are strongly associated with each sentiment"""
    df = load_model_predictions(model, domain)
    if df is None:
        return
    
    # Combine sentences for each sentiment
    sentiment_texts = {}
    for sentiment in df['predicted_polarity'].unique():
        sentiment_texts[sentiment] = ' '.join(df[df['predicted_polarity'] == sentiment]['sentence'])
    
    # Tokenize and get word frequencies
    stop_words = set(stopwords.words('english'))
    sentiment_words = {}
    
    for sentiment, text in sentiment_texts.items():
        # Tokenize and filter out stop words
        tokens = word_tokenize(text.lower())
        filtered_tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
        
        # Get word frequencies
        word_freq = Counter(filtered_tokens)
        sentiment_words[sentiment] = word_freq
    
    # Find distinctive words for each sentiment
    all_words = set()
    for word_freq in sentiment_words.values():
        all_words.update(word_freq.keys())
    
    distinctive_words = {}
    for sentiment, word_freq in sentiment_words.items():
        # Calculate distinctiveness score
        words_with_scores = []
        for word in word_freq:
            # Count in this sentiment
            count_in_sentiment = word_freq[word]
            
            # Count in other sentiments
            count_in_others = sum(other_freq.get(word, 0) for other_sentiment, other_freq in sentiment_words.items() 
                               if other_sentiment != sentiment)
            
            # Calculate distinctiveness score (higher is more distinctive)
            distinctiveness = count_in_sentiment / (count_in_others + 1)  # +1 to avoid division by zero
            
            words_with_scores.append((word, count_in_sentiment, distinctiveness))
        
        # Sort by distinctiveness and get top words
        distinctive_words[sentiment] = sorted(words_with_scores, key=lambda x: x[2], reverse=True)[:20]
    
    # Print distinctive words
    print(f"\n{model.upper()} - {domain.upper()} - Distinctive Words by Sentiment:")
    for sentiment, words in distinctive_words.items():
        print(f"\n  {sentiment.upper()}:")
        for word, count, score in words[:10]:  # Show top 10
            print(f"    {word}: count={count}, distinctiveness={score:.2f}")
    
    return distinctive_words

def analyze_confusion_patterns(domain, models):
    """Analyze patterns in model disagreements"""
    disagreements = analyze_common_errors(domain, models)
    if disagreements is None or len(disagreements) == 0:
        return
    
    print(f"\nAnalyzing confusion patterns for {domain.upper()} domain:")
    
    # Count disagreement patterns
    patterns = []
    for _, row in disagreements.iterrows():
        pattern = {}
        for m in models:
            if m in row:
                pattern[m] = row[m]
        patterns.append(tuple(sorted(pattern.items())))
    
    pattern_counts = Counter(patterns)
    
    # Print common disagreement patterns
    print("\nCommon disagreement patterns:")
    for pattern, count in pattern_counts.most_common(5):
        pattern_str = " vs ".join([f"{m}: {s}" for m, s in pattern])
        print(f"  {pattern_str}: {count} instances")
    
    # Analyze by aspect
    aspect_counts = disagreements['aspect'].value_counts()
    print("\nTop aspects with model disagreements:")
    for aspect, count in aspect_counts.head(10).items():
        print(f"  {aspect}: {count} instances")
    
    return disagreements

def generate_error_report(domains, models):
    """Generate a comprehensive error analysis report"""
    os.makedirs("outputs/analysis", exist_ok=True)
    
    for domain in domains:
        print(f"\n{'='*70}")
        print(f"ERROR ANALYSIS FOR {domain.upper()} DOMAIN")
        print(f"{'='*70}")
        
        # Analyze prediction patterns
        for model in models:
            analyze_errors_by_sentiment(domain, model)
        
        # Analyze aspects
        aspect_analyses = {}
        for model in models:
            aspect_analyses[model] = analyze_aspects(domain, model)
        
        # Analyze sentence length effects
        for model in models:
            analyze_sentence_length(domain, model)
        
        # Analyze keywords
        for model in models:
            analyze_keywords(domain, model)
        
        # Analyze confusion patterns
        analyze_confusion_patterns(domain, models)
        
        # Save full report
        with open(f"outputs/analysis/{domain}_error_analysis.txt", "w") as f:
            # Redirect print output to file
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            
            print(f"ERROR ANALYSIS REPORT FOR {domain.upper()} DOMAIN")
            print(f"{'='*70}")
            
            # Re-run analyses with output to file
            for model in models:
                analyze_errors_by_sentiment(domain, model)
                analyze_aspects(domain, model)
                analyze_sentence_length(domain, model)
                analyze_keywords(domain, model)
            
            analyze_confusion_patterns(domain, models)
            
            # Restore stdout
            sys.stdout = original_stdout
        
        print(f"\nFull error analysis report saved to outputs/analysis/{domain}_error_analysis.txt")

if __name__ == "__main__":
    domains = ['laptop', 'restaurant']
    models = ['svm', 'lr', 'bert', 'hybrid']
    
    generate_error_report(domains, models)