# Aspect‑Based Sentiment Analysis (ABSA) Project

Aspect‑Based Sentiment Analysis (ABSA) drills down below overall document sentiment to tell **what** people feel (**sentiment polarity**) about **which parts** of a product or service (**aspects**).  
This repository implements and compares four approaches for ABSA on the SemEval‑2014 Task 4 laptop & restaurant review datasets.

![ABSA Visualization Example](outputs/analysis/example_visualization.png)

---

## 1  Models Implemented

| Model | Brief Description |
|-------|-------------------|
| **Support Vector Machine (SVM)** | Classic linear‑kernel SVM on engineered linguistic features |
| **Logistic Regression (LR)** | Fast linear classifier baseline |
| **BERT** | Transformer fine‑tuned end‑to‑end for aspect‑level polarity |
| **Hybrid BERT** | BERT sentence embeddings + engineered features fed to an SVM |

---

## 2  Key Features

- End‑to‑end preprocessing pipeline (XML → clean tokenised CSV)
- Feature engineering<br> • TF‑IDF unigrams/bigrams • SentiWordNet scores • Dependency‑parse relations  
- Unified training / evaluation runners with **JSON metric logs**
- Rich error‑analysis module producing:<br> • Sentiment‑distribution pies • Aspect frequency bars • Word clouds • Model‑disagreement heat‑maps
- **Auto‑generated interactive HTML report** per domain

---

## 3  Project Structure

\`\`\`
absa_project/
├─ data/                    # Datasets
│  ├─ raw/                  # Original SemEval XML
│  └─ processed/            # Train/val/test CSV
│
├─ preprocessing/
│  ├─ preprocess.py         # Build train/val/test splits
│  └─ preprocesstest.py    # For unseen test sets
│
├─ models/
│  ├─ traditional/
│  │  ├─ SVMModel.py
│  │  └─ train_modelLR.py
│  └─ bert/
│     ├─ bert_classifier.py
│     └─ hybrid_classifier.py
│
├─ runners/                 # CLI entrypoints
│  ├─ run_bert_classifier.py
│  └─ run_hybrid_classifier.py
│
├─ utils/
│  └─ aspect_extractor.py
│
├─ outputs/
│  ├─ traditional/          # *.csv predictions & *.json metrics
│  ├─ bert/
│  ├─ hybrid/
│  └─ analysis/             # All visualisations & reports
│
└─ README.md
\`\`\`

---

## 4  Dataset

*SemEval‑2014 Task 4*  
- **Laptop** domain reviews  
- **Restaurant** domain reviews  

Each sentence is annotated with one or more \`(aspect, sentiment)\` pairs where sentiment ∈ {**positive**, **negative**, **neutral**}.

---

## 5  Getting Started

### 5.1  Prerequisites

| Package | Version (or newer) |
|---------|--------------------|
| Python | 3.8+ |
| PyTorch | 1.8 |
| transformers | 4.\* |
| scikit‑learn | 0.24 |
| nltk, spacy, seaborn, tabulate, wordcloud, colorama |

### 5.2  Installation

\`\`\`bash
https://github.com/Sanket01234/INLPProject.git
cd INLPProject/

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLP resources
python3 -m nltk.downloader punkt averaged_perceptron_tagger sentiwordnet stopwords
python3 -m spacy download en_core_web_sm
\`\`\`

---

## 6  Pipeline Execution

### 6.1  Pre‑process Data

\`\`\`bash
python3 preprocessing/preprocess.py
python3 preprocessing/preprocesstest.py
\`\`\`

### 6.2  Train / Evaluate Models

\`\`\`bash
# Traditional models
python3 models/traditional/SVMModel.py
python3 models/traditional/train_modelLR.py

# BERT fine‑tuning
python3 runners/run_bert_classifier.py

# Hybrid model
python3 runners/run_hybrid_classifier.py
\`\`\`

### 6.3  Analyse Results

\`\`\`bash
# Compare metrics
python3 compare_models.py  

# Generate detailed error analysis + HTML report
python3 error_analysis.py
\`\`\`

---

## 7  Results & Visualisation

\`outputs/analysis/\` contains

- **Performance metrics** (Accuracy, Precision, Recall, F1‑macro)
- **Confusion matrices**
- Sentiment‑distribution **pie charts**
- **Aspect frequency** horizontal bars
- **Word clouds** for distinctive tokens
- Interactive **HTML reports** (\`domain_html_report/index.html\`)

---

## 8  Sample Findings

| Observation | Insight |
|-------------|---------|
| 🚀 **Traditional (SVM/LR)** | Very fast training & interpretable weights, but lower recall on minority classes |
| 🤖 **BERT** | Highest accuracy & F1; captures contextual nuances (e.g., “keyboard *light*”) |
| 🔀 **Hybrid** | Slightly below BERT’s performance yet 2‑3× faster inference—good trade‑off for production |

---

## 9  License

This project is released under the [MIT License](LICENSE).

---

## 10  Acknowledgements

- **SemEval‑2014 Task 4** organisers  
- [Hugging Face Transformers](https://huggingface.co/)  
- NLTK & spaCy teams

---

## 11  Contributors

- **Nikhil Singh, Sanket Madaan, Digvijay Singh Rathore** – initial implementation, model training, visualisation