import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, confusion_matrix)
from joblib import dump, load
from aspect_extractor import AspectExtractor

class ABSA_Classifier:
    def __init__(self, domain):
        self.domain = domain
        self.aspect_extractor = AspectExtractor()
        self.tfidf = load(f"{domain}_tfidf_vectorizer.joblib")
        self.bow = load(f"{domain}_bow_vectorizer.joblib")
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
        # Load data
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        
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
        
        # Save model and metrics
        dump(self.clf, f"{self.domain}_lr_classifier.joblib")
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def _evaluate(self, X, y_true, split_name):
        """Evaluate model performance"""
        y_pred = self.clf.predict(X)
        return {
            f'{split_name}_accuracy': accuracy_score(y_true, y_pred),
            f'{split_name}_precision_macro': precision_score(y_true, y_pred, average='macro'),
            f'{split_name}_recall_macro': recall_score(y_true, y_pred, average='macro'),
            f'{split_name}_f1_macro': f1_score(y_true, y_pred, average='macro'),
            f'{split_name}_confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
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
        test_df = pd.read_csv(test_csv)
        X_test = self.prepare_features(test_df)
        y_pred = self.clf.predict(X_test)
        test_df['predicted_polarity'] = [self.reverse_label_map[p] for p in y_pred]
        return test_df

if __name__ == "__main__":
    domains = ['laptop', 'restaurant']
    
    # Train domain-specific models
    for domain in domains:
        print(f"Training {domain} model...")
        absa = ABSA_Classifier(domain)
        metrics = absa.train(f"{domain}_train_features.csv", f"{domain}_val_features.csv")
        print(f"{domain} metrics:", json.dumps(metrics, indent=2))
    
    # Predict for each domain's test data
    for domain in domains:
        # Load test data WITH EXTRACTED ASPECTS from preprocessing
        test_df = pd.read_csv(f"{domain}_test_features_unlabeled.csv")
        
        # Load model and features
        absa = ABSA_Classifier(domain)
        absa.clf = load(f"{domain}_lr_classifier.joblib")
        X_test = np.load(f"{domain}_X_test_unlabeled.npy")
        
        # Predict polarities
        y_pred = absa.clf.predict(X_test)
        
        # Add predictions to dataframe
        test_df['predicted_polarity'] = [absa.reverse_label_map[p] for p in y_pred]
        
        # Save predictions WITH EXTRACTED ASPECTS
        test_df[['sentence', 'aspect', 'predicted_polarity']].to_csv(
            f"{domain}_test_predictions.csv", 
            index=False
        )
        print(f"{domain} predictions saved with extracted aspects.")
