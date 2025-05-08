from HybridBERTClassifier import HybridABSA_Classifier
import pandas as pd
import os

def main():
    # Paths to preprocessed data
    train_csv = "bert_data/restaurant_train_features.csv"  # Example path
    val_csv = "bert_data/restaurant_val_features.csv"
    output_dir = "hybrid_model_output_restaurant"

    # Initialize Hybrid Classifier (with TF-IDF if needed)
    hybrid_classifier = HybridABSA_Classifier(
        model_name="bert-base-uncased",
        num_labels=3,
        tfidf_dim=200,  
        use_attention=True
    )

    # Train the model
    metrics = hybrid_classifier.train(
        train_csv=train_csv,
        val_csv=val_csv,
        output_dir=output_dir,
        batch_size=16,
        epochs=4,
        learning_rate=2e-5
    )

    # Optional: Load the best model and predict
    hybrid_classifier.load_model(os.path.join(output_dir, "best_model"))
    test_df = hybrid_classifier.predict_from_test_data(
        "bert_data/restaurant_test_features.csv",
        output_csv="hybrid_model_output_restaurant/predictions.csv"
    )

if __name__ == "__main__":
    main()
