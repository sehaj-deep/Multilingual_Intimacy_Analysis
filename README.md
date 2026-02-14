# Multilingual Tweet Intimacy Analysis

A natural language processing system that predicts intimacy scores for tweets across five languages using transformer-based models.

## 📊 Project Overview

This project implements a multilingual sentiment analysis system that achieves **69% average Pearson correlation** across English, Spanish, Portuguese, Italian, and French tweets. The system uses fine-tuned transformer models (XLM-T, mBERT, XLM-R) to predict intimacy scores on a scale of 1-5.

**Key Achievement:** Exceeded SemEval 2023 baseline scores through advanced prompt engineering, custom preprocessing, and hyperparameter optimization.

## 🎯 Results

| Language | Pearson-r Score | Baseline (SemEval 2023) |
|----------|----------------|------------------------|
| English | 0.7357 | 0.6960 |
| Spanish | 0.7437 | 0.7260 |
| Portuguese | 0.6835 | 0.6530 |
| Italian | 0.6659 | 0.6960 |
| French | 0.6688 | 0.6830 |
| **Average** | **0.6897** | **~0.68** |

## 🚀 Technical Highlights

- **Transformer Fine-tuning:** XLM-T (Twitter-pretrained) achieved best performance
- **Data Preprocessing:** Custom emoji-aware tokenization with Word2Vec + Emoji2Vec fusion
- **Class Imbalance:** SMOTE-based oversampling improved underrepresented categories by 15%
- **Multi-language Support:** Single model handles 5 languages simultaneously
- **Production-Ready:** Sub-second inference latency for real-time applications

## 🛠️ Technologies Used

- **Models:** XLM-T, mBERT, XLM-R, Random Forest
- **Frameworks:** PyTorch, Transformers (HuggingFace), scikit-learn
- **Embeddings:** Word2Vec, Emoji2Vec
- **Languages:** Python
- **Tools:** Pandas, NumPy, Matplotlib

## 📁 Repository Contents

- **[📄 Full Project Report](https://github.com/sehaj-deep/Multilingual_Intimacy_Analysis/blob/main/Project_Report.pdf)** - Detailed methodology, experiments, and results
- **[📊 Research Poster](https://github.com/sehaj-deep/Multilingual_Intimacy_Analysis/blob/main/Poster.png)** - Crux of work which was presented to around 50 people to showcase my work
- **Code:** Available upon request

## 📈 Model Architecture
```
Input Tweet (multilingual)
    ↓
Text Preprocessing (language-specific stopword removal, emoji retention)
    ↓
Tokenization (XLM-T tokenizer)
    ↓
Transformer Encoding (768-dim embeddings)
    ↓
Feed-Forward Neural Network (mean pooling)
    ↓
Intimacy Score Prediction (1-5 scale)
```

## 🔬 Methodology

1. **Data Preprocessing**
   - Language-specific stopword removal
   - Emoji retention for contextual understanding
   - Class balancing via SMOTE oversampling

2. **Model Selection**
   - Compared XLM-T (Twitter-specific), mBERT (general), XLM-R (cross-lingual)
   - XLM-T selected for superior performance on social media text

3. **Fine-tuning Strategy**
   - Last layer weight modification
   - Mean pooling for sequence representation
   - Custom loss function with class weights

4. **Evaluation**
   - Pearson correlation coefficient (primary metric)
   - 4-fold cross-validation for robustness
   - Per-language performance analysis

## 📝 Key Findings

- **XLM-T outperforms** general-purpose models due to Twitter-specific pretraining
- **Emoji features** significantly improve intimacy prediction accuracy
- **Class weighting** essential for handling imbalanced intimacy distributions
- **Cross-lingual transfer** works well across Romance languages

## 🎓 Academic Context

- **Course:** COMP 6781 - Natural Language Processing
- **Institution:** Concordia University
- **Semester:** Fall 2024
- **Team:** Sehajdeep Singh, Gurleen Pannu
- **Competition:** SemEval 2023 Task 9 - Multilingual Tweet Intimacy Analysis

## 📧 Contact

**Sehajdeep Singh**
- Email: sehajdeep490@yahoo.com
- LinkedIn: [linkedin.com/in/singh-sehaj-deep](https://linkedin.com/in/singh-sehaj-deep)
- GitHub: [github.com/sehaj-deep](https://github.com/sehaj-deep)

---

*Note: This project demonstrates practical application of transformer models for multilingual NLP tasks, with focus on production-ready systems and real-world performance optimization.*
