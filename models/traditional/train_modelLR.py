import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, confusion_matrix)
from joblib import dump, load

# Try different import paths for AspectExtractor
try:
    from models.utils.aspect_extractor import AspectExtractor
except ModuleNotFoundError:
    try:
        # Try relative import
        from ..utils.aspect_extractor import AspectExtractor
    except (ImportError, ValueError):
        try:
            # Try direct import if in same directory
            from aspect_extractor import AspectExtractor
        except ModuleNotFoundError:
            print("WARNING: Could not import AspectExtractor. Some functionality will be limited.")
            # Define a minimal AspectExtractor class to avoid errors
            class AspectExtractor:
                def __init__(self):
                    pass
                def extract_aspects(self, sentence, domain):
                    return ['general']

class ABSA_Classifier:
    def __init__(self, domain):
        self.domain = domain
        self.aspect_extractor = AspectExtractor()
        
        # Try multiple possible locations for the vectorizer files
        tfidf_paths = [
            f"{domain}_tfidf_vectorizer.joblib",
            f"data/processed/{domain}_tfidf_vectorizer.joblib", 
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                       f"data/processed/{domain}_tfidf_vectorizer.joblib")
        ]
        
        bow_paths = [
            f"{domain}_bow_vectorizer.joblib",
            f"data/processed/{domain}_bow_vectorizer.joblib",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                       f"data/processed/{domain}_bow_vectorizer.joblib")
        ]
        
        # Try each possible path for tfidf vectorizer
        for path in tfidf_paths:
            try:
                print(f"Trying to load TF-IDF vectorizer from: {path}")
                self.tfidf = load(path)
                print(f"Successfully loaded TF-IDF vectorizer from {path}")
                break
            except FileNotFoundError:
                continue
        else:
            raise FileNotFoundError(f"Could not find TF-IDF vectorizer for {domain} in any location")
        
        # Try each possible path for bow vectorizer
        for path in bow_paths:
            try:
                print(f"Trying to load BOW vectorizer from: {path}")
                self.bow = load(path)
                print(f"Successfully loaded BOW vectorizer from {path}")
                break
            except FileNotFoundError:
                continue
        else:
            raise FileNotFoundError(f"Could not find BOW vectorizer for {domain} in any location")
        
        self.clf = None
        self.label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def prepare_features(self, df):
        """Prepare feature matrix from dataframe"""
        tfidf_features = self.tfidf.transform(df['text'])
        bow_features = self.bow.transform(df['text'])
        return np.hstack([
            tfidf_features.toarray(),
            bow_features.toarray(),
            df[['lexicon_pos', 'lexicon_neg']].values
        ])
    
    def train(self, train_csv, val_csv):
        """Train the classifier"""
        # Try multiple possible locations for CSV files
        train_paths = [
            train_csv,
            f"data/processed/{train_csv}",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        f"data/processed/{train_csv}")
        ]
        
        val_paths = [
            val_csv,
            f"data/processed/{val_csv}",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        f"data/processed/{val_csv}")
        ]
        
        # Find train CSV
        for path in train_paths:
            if os.path.exists(path):
                print(f"Loading training data from: {path}")
                train_df = pd.read_csv(path)
                break
        else:
            raise FileNotFoundError(f"Could not find training data: {train_csv}")
        
        # Find validation CSV
        for path in val_paths:
            if os.path.exists(path):
                print(f"Loading validation data from: {path}")
                val_df = pd.read_csv(path)
                break
        else:
            raise FileNotFoundError(f"Could not find validation data: {val_csv}")
        
        # Prepare features
        X_train = self.prepare_features(train_df)
        X_val = self.prepare_features(val_df)
        y_train = train_df['polarity'].map(self.label_map)
        y_val = val_df['polarity'].map(self.label_map)
        
        # Train model
        self.clf = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        self.clf.fit(X_train, y_train)
        
        # Evaluate
        metrics = {
            'training': self._evaluate(X_train, y_train, 'training'),
            'validation': self._evaluate(X_val, y_val, 'validation')
        }
        
        # Create output directory if it doesn't exist
        os.makedirs("outputs/traditional", exist_ok=True)
        
        # Save model and metrics
        dump(self.clf, f"outputs/traditional/{self.domain}_lr_classifier.joblib")
        with open(f"outputs/traditional/{self.domain}_lr_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def _evaluate(self, X, y_true, split_name):
        """Evaluate model performance with per-class metrics"""
        y_pred = self.clf.predict(X)
        
        # Calculate per-class precision and recall
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # Create a dictionary of per-class metrics
        per_class_metrics = {}
        for i, label in enumerate(self.reverse_label_map.values()):
            per_class_metrics[label] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i])
            }
        
        return {
            f'{split_name}_accuracy': accuracy_score(y_true, y_pred),
            f'{split_name}_precision_macro': precision_score(y_true, y_pred, average='macro'),
            f'{split_name}_recall_macro': recall_score(y_true, y_pred, average='macro'),
            f'{split_name}_f1_macro': f1_score(y_true, y_pred, average='macro'),
            f'{split_name}_confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            f'{split_name}_per_class_metrics': per_class_metrics
        }
    
    def predict_from_text(self, sentences, domain):
        """Predict aspects and polarities for raw sentences"""
        if not self.clf:
            raise ValueError("Model not trained. Call train() first.")
        
        results = []
        for sentence in sentences:
            # Extract aspects
            aspects = self.aspect_extractor.extract_aspects(sentence, domain)
            
            if not aspects:
                # Fallback to general sentiment
                aspects = ['general']
            
            # Predict polarity for each aspect
            for aspect in aspects:
                # Create aspect-sentence pair
                text = f"{sentence} {aspect}"
                
                # Prepare features
                tfidf_feats = self.tfidf.transform([text])
                bow_feats = self.bow.transform([text])
                
                # Get lexicon features (implement as needed)
                lexicon_feats = {'lexicon_pos': 0, 'lexicon_neg': 0}  # Placeholder
                
                # Combine features
                X = np.hstack([
                    tfidf_feats.toarray(),
                    bow_feats.toarray(),
                    np.array([lexicon_feats['lexicon_pos'], lexicon_feats['lexicon_neg']])
                ])
                
                # Predict
                polarity_idx = self.clf.predict(X)[0]
                polarity = self.reverse_label_map[polarity_idx]
                
                results.append({
                    'sentence': sentence,
                    'aspect': aspect,
                    'predicted_polarity': polarity
                })
        
        return pd.DataFrame(results)
    
    def predict_from_test_data(self, test_csv):
        """Predict polarities for preprocessed test data with extracted aspects"""
        # Try multiple possible locations for test CSV
        test_paths = [
            test_csv,
            f"data/processed/{test_csv}",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                       f"data/processed/{test_csv}")
        ]
        
        # Find test CSV
        for path in test_paths:
            if os.path.exists(path):
                print(f"Loading test data from: {path}")
                test_df = pd.read_csv(path)
                break
        else:
            raise FileNotFoundError(f"Could not find test data: {test_csv}")
            
        X_test = self.prepare_features(test_df)
        y_pred = self.clf.predict(X_test)
        test_df['predicted_polarity'] = [self.reverse_label_map[p] for p in y_pred]
        return test_df

if __name__ == "__main__":
    # Create output directory
    os.makedirs("outputs/traditional", exist_ok=True)
    
    domains = ['laptop', 'restaurant']
    
    # Train domain-specific models
    for domain in domains:
        print(f"Training {domain} model...")
        try:
            absa = ABSA_Classifier(domain)
            metrics = absa.train(f"{domain}_train_features.csv", f"{domain}_val_features.csv")
            print(f"{domain} metrics:", json.dumps(metrics, indent=2))
        except Exception as e:
            print(f"Error training {domain} model: {e}")
    
    # Predict for each domain's test data
    for domain in domains:
        try:
            # Load test data WITH EXTRACTED ASPECTS from preprocessing
            test_file = f"{domain}_test_features_unlabeled.csv"
            
            # Load model and initialize classifier
            absa = ABSA_Classifier(domain)
            
            # Try multiple possible locations for the model file
            model_paths = [
                f"{domain}_lr_classifier.joblib",
                f"outputs/traditional/{domain}_lr_classifier.joblib",
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           f"outputs/traditional/{domain}_lr_classifier.joblib")
            ]
            
            # Load the model
            for path in model_paths:
                if os.path.exists(path):
                    print(f"Loading model from: {path}")
                    absa.clf = load(path)
                    break
            else:
                raise FileNotFoundError(f"Could not find model file for {domain}")
            
            # Load test data and make predictions
            test_df = absa.predict_from_test_data(test_file)
            
            # Save predictions WITH EXTRACTED ASPECTS
            output_path = f"outputs/traditional/{domain}_lr_test_predictions.csv"
            test_df[['sentence', 'aspect', 'predicted_polarity']].to_csv(output_path, index=False)
            print(f"{domain} predictions saved to {output_path}")
        except Exception as e:
            print(f"Error predicting for {domain}: {e}")