# SVMModel.py
import pandas as pd
import numpy as np
import json
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, confusion_matrix)
from joblib import dump, load

class SVM_Classifier:
    def __init__(self, domain):
        self.domain = domain
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
        """Train the SVM classifier"""
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        
        X_train = self.prepare_features(train_df)
        X_val = self.prepare_features(val_df)
        y_train = train_df['polarity'].map(self.label_map)
        y_val = val_df['polarity'].map(self.label_map)
        
        # Using CalibratedClassifierCV for probability estimates
        self.clf = CalibratedClassifierCV(
            LinearSVC(
                class_weight='balanced',
                max_iter=10000,
                random_state=42,
                dual=False
            ),
            method='sigmoid'
        )
        self.clf.fit(X_train, y_train)
        
        metrics = {
            'training': self._evaluate(X_train, y_train, 'training'),
            'validation': self._evaluate(X_val, y_val, 'validation')
        }
        
        # Save model and metrics
        dump(self.clf, f"{self.domain}_svm_classifier.joblib")
        with open(f"{self.domain}_svm_metrics.json", "w") as f:
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
    
    def predict_from_test_data(self, test_csv):
        """Predict polarities for preprocessed test data"""
        test_df = pd.read_csv(test_csv)
        X_test = self.prepare_features(test_df)
        y_pred = self.clf.predict(X_test)
        test_df['predicted_polarity'] = [self.reverse_label_map[p] for p in y_pred]
        return test_df

if __name__ == "__main__":
    domains = ['laptop', 'restaurant']
    
    # Train SVM models for each domain
    for domain in domains:
        print(f"Training {domain} SVM model...")
        svm_classifier = SVM_Classifier(domain)
        metrics = svm_classifier.train(
            f"{domain}_train_features.csv",
            f"{domain}_val_features.csv"
        )
        print(f"{domain} SVM metrics:", json.dumps(metrics, indent=2))
    
    # Generate predictions for each domain
    for domain in domains:
        svm_classifier = SVM_Classifier(domain)
        svm_classifier.clf = load(f"{domain}_svm_classifier.joblib")
        
        test_df = pd.read_csv(f"{domain}_test_features_unlabeled.csv")
        predictions = svm_classifier.predict_from_test_data(f"{domain}_test_features_unlabeled.csv")
        
        # Save predictions with aspects
        predictions[['sentence', 'aspect', 'predicted_polarity']].to_csv(
            f"{domain}_svm_test_predictions.csv",
            index=False
        )
        print(f"{domain} SVM predictions saved.")