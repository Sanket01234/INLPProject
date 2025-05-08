import argparse
import os
import logging
from BERTModel import ABSA_BERT_Classifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run BERT classifier for ABSA')
    parser.add_argument('--domains', nargs='+', default=['laptop', 'restaurant'], 
                        help='Domains to process (laptop, restaurant, or both)')
    parser.add_argument('--data_dir', default='bert_data', 
                        help='Directory with preprocessed data')
    parser.add_argument('--output_dir', default='bert_results', 
                        help='Directory to save model and results')
    parser.add_argument('--model', default='bert-base-uncased', 
                        help='BERT model to use')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=4, 
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, 
                        help='Learning rate')
    parser.add_argument('--mode', choices=['train', 'test', 'both'], default='both',
                        help='Run mode: train, test, or both')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    for domain in args.domains:
        domain_output = os.path.join(args.output_dir, domain)
        os.makedirs(domain_output, exist_ok=True)
        
        # Initialize BERT classifier
        bert_classifier = ABSA_BERT_Classifier(model_name=args.model)
        
        # Train the model
        if args.mode in ['train', 'both']:
            logger.info(f"Training {domain} BERT model...")
            train_csv = os.path.join(args.data_dir, f"{domain}_train_features.csv")
            val_csv = os.path.join(args.data_dir, f"{domain}_val_features.csv")
            
            metrics = bert_classifier.train(
                train_csv=train_csv,
                val_csv=val_csv,
                output_dir=domain_output,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.lr
            )
            
            logger.info(f"{domain} training complete. Metrics saved to {domain_output}")
        
        # Test the model
        if args.mode in ['test', 'both']:
            logger.info(f"Testing {domain} model...")
            
            # Load the best model
            model_path = os.path.join(domain_output, "best_model")
            
            # If we're only testing, we need to load the model
            if args.mode == 'test':
                bert_classifier.load_model(model_path)
            
            test_csv = os.path.join(args.data_dir, f"{domain}_test_features.csv")
            output_csv = os.path.join(domain_output, f"{domain}_bert_predictions.csv")
            
            predictions = bert_classifier.predict_from_test_data(
                test_csv=test_csv,
                output_csv=output_csv
            )
            
            logger.info(f"{domain} predictions saved to {output_csv}")

if __name__ == "__main__":
    main()
