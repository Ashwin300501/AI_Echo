# 🧠 AI Echo — ChatGPT Review Sentiment Dashboard

AI Echo is a **Streamlit-based interactive dashboard** that analyzes ChatGPT user reviews using a **fine-tuned DistilBERT model**.  
It predicts **Positive**, **Neutral**, or **Negative** sentiment directly from review text and visualizes insights across time, user groups, and versions.

---

## 🚀 Project Overview

### 🎯 Objective
To build a complete **end-to-end NLP pipeline** for ChatGPT user review analysis:
1. Clean and preprocess textual reviews  
2. Train a Transformer-based model for sentiment classification  
3. Evaluate performance using standard metrics and confusion matrix  
4. Deploy a Streamlit dashboard for real-time predictions and insights  

---

## 🧩 Architecture

```text
📁 AI_Echo/
│
├── data/
│   ├── chatgpt_style_reviews_dataset.xlsx         ← raw dataset
│   └── clean/reviews_clean.parquet                ← preprocessed dataset
│
├── eda/
│   └── plots/                                    ← saved EDA visualizations
│
├── models/
│   └── distilbert_sentiment/                      ← fine-tuned DistilBERT model
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer.json / tokenizer_config.json
│       └── special_tokens_map.json
│
├── artifacts/
│   ├── phase1_config.json
│   ├── cm_distilbert_binary.png                   ← confusion matrix
│   └── phase2_metrics_distilbert_binary.json      ← model performance metrics
│
├── app.py                                         ← Streamlit app (model-only)
├── README.md
└──requirements.txt
```

---

## ⚙️ Workflow Summary

### **1️⃣ Data Preprocessing**
- Removed punctuation, URLs, mentions, emojis, and stopwords  
- Applied lemmatization using NLTK  
- Created derived features:
  - `review_length`
  - `review_tokens`
  - `month`, `week`, `verified_purchase`, etc.  
- Saved cleaned dataset as `reviews_clean.parquet`

### **2️⃣ EDA**
Visualized rating distributions and patterns:
- Rating frequency  
- Word clouds for positive/negative reviews  
- Average rating over time  
- Verified vs Non-Verified comparison  
- Location and version insights  

### **3️⃣ Model Training (DistilBERT)**
- Used HuggingFace `transformers` for fine-tuning  
- Removed oversampling for neutral reviews  
- Dataset balanced only between positive and negative classes where necessary  
- Config:
  ```python
  learning_rate = 2e-5
  batch_size = 16
  epochs = 3
  model = "distilbert-base-uncased"
  ```
- Metrics used:
  - Accuracy  
  - F1-score (Weighted & Binary)  
  - Confusion Matrix  

### **4️⃣ Dashboard (Streamlit)**
- **Tab 1: 💬 Sentiment Prediction**
  - Enter custom reviews and get real-time predictions from DistilBERT  
  - Visualize model confidence scores  
  - Displays overall sentiment distribution (model predictions)

- **Tab 2: 📊 Key Questions for Sentiment Analysis**
  - Overall sentiment distribution  
  - Sentiment vs rating mismatch  
  - Word clouds for each sentiment class  
  - Sentiment trends over time  
  - Verified user sentiment patterns  
  - Review length vs sentiment  
  - Location and platform breakdowns  
  - Version impact on sentiment  
  - Common themes in negative reviews  

---

## 🧠 Model Details

| Component | Description |
|------------|-------------|
| **Base Model** | DistilBERT (base uncased) |
| **Task** | Multi-class Sentiment Classification (3 classes) |
| **Labels** | Negative / Neutral / Positive |
| **Tokenizer** | DistilBertTokenizerFast |
| **Optimizer** | AdamW |
| **Evaluation Metric** | F1-weighted, Accuracy |

---

## 🧾 Example Outputs

### 🔹 Confusion Matrix
Saved at `artifacts\cm_distilbert_3class.png`

### 🔹 Evaluation Metrics
| Metric | Score |
|---------|-------|
| **Accuracy** | ~0.44 |
| **Weighted F1-Score** | ~0.38 |

Confusion Matrix example:  
![Confusion Matrix](artifacts\cm_distilbert_3class.png)

---

## 📦 Artifacts & Outputs

| File | Description |
|------|--------------|
| `data/clean/reviews_clean.parquet` | Cleaned review data |
| `artifacts/phase1_config.json` | Phase 1 configuration metadata |
| `artifacts/phase2_metrics_distilbert_binary.json` | Model performance metrics |
| `artifacts/cm_distilbert_binary.png` | Confusion matrix visualization |
| `eda/plots/` | Word clouds, rating trends, etc. |

---

## 🚫 Model Files Note

The trained DistilBERT weights are **not included** due to GitHub’s 100 MB limit.

You can load the model from Hugging Face:
```
Ashwin20015/ai-echo-distilbert-sentiment
```
Or place your own trained model under:
```
models/distilbert_sentiment/
```
3. The app will automatically detect and use it for inference.

---

## 💡 Key Insights

- Positive reviews emphasize *ease of use*, *creativity*, and *accuracy*.  
- Negative reviews frequently mention *hallucinations*, *incorrect answers*, and *slow response*.  
- Verified users tend to leave more positive reviews.  
- Sentiment improves with later ChatGPT versions.  
 
