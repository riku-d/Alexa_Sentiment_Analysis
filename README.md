# Amazon Alexa Sentiment Analysis (SVM)

An end-to-end **Machine Learning + Data Science project** that analyzes customer reviews of Amazon Alexa devices and predicts sentiment in real time using a **Support Vector Machine (SVM)** model.

---
## 🌐 Live Demo

[![Open App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit)](https://alexasentimentanalysis-ml.streamlit.app/)

---
## 🚀 Live Features

* 🔍 Real-time sentiment prediction (Positive / Negative / Neutral)
* ⚡ Pre-trained model — instant predictions (no retraining needed)
* 📊 Interactive dashboard with EDA & visualizations
* 🎯 Confidence-based neutral sentiment detection
* 📈 Full model evaluation (ROC, Confusion Matrix, F1-score)
* 🧠 Step-by-step ML pipeline explanation
* 🎨 Modern UI built with Streamlit

---

## 🧠 Problem Statement

Customer reviews often contain mixed or unclear sentiment.
The goal is to:

* Classify reviews as **Positive or Negative**
* Handle **ambiguous/neutral cases**
* Build an **interpretable ML system** for real-world use

---

## ⚙️ Tech Stack

* **Language:** Python
* **Frontend:** Streamlit
* **ML:** Scikit-learn
* **NLP:** NLTK
* **Visualization:** Plotly, Matplotlib, Seaborn
* **Data Handling:** Pandas, NumPy

---

## 🔄 Machine Learning Pipeline

### 1️⃣ Text Preprocessing

* Lowercasing
* Regex cleaning (remove special characters)
* Stopword removal (except negations)
* Porter Stemming
* Negation handling → `not_good`, `not_buy`

---

### 2️⃣ Feature Engineering

* **TF-IDF Vectorizer**

  * `max_features = 3000`
  * `ngram_range = (1,2)` (unigrams + bigrams)
  * `sublinear_tf = True`

---

### 3️⃣ Handling Imbalance

* **SMOTE (Synthetic Minority Oversampling)**
* Ensures balanced training data

---

### 4️⃣ Feature Scaling

* **MaxAbsScaler**
* Preserves sparsity (important for TF-IDF)

---

### 5️⃣ Model Training

* **Linear SVM (LinearSVC)**
* Wrapped with **CalibratedClassifierCV**

  * Enables probability predictions

---

### 6️⃣ Threshold Optimization

* Custom decision threshold tuned using **F1-score**
* Improves minority class (Negative) detection

---

## 🧠 Model Decision Logic

Instead of strict binary classification:

```
if |Positive - Negative| < 0.2 → Neutral
elif Positive > Negative → Positive
else → Negative
```

👉 This improves real-world usability

---

## 📊 Evaluation Metrics

* Accuracy
* Precision / Recall
* F1 Score (Macro & Weighted)
* ROC-AUC
* Confusion Matrix

---

## 📈 Visualizations

* Rating distribution
* Sentiment distribution
* Word clouds
* Review length analysis
* ROC curve
* Learning curve
* SVM decision boundary (demo)

---

## 🔍 Example Predictions

| Review                            | Prediction |
| --------------------------------- | ---------- |
| "Amazing product, love it!"       | ✅ Positive |
| "Terrible device, waste of money" | ❌ Negative |
| "Works fine, nothing special"     | 😐 Neutral |

---

## 🖥️ Running the Project

### 🔹 1. Clone repository

```bash
git clone https://github.com/your-username/Alexa_Sentiment_Analysis.git
cd Alexa_Sentiment_Analysis
```

---

### 🔹 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 🔹 3. Run Streamlit app

```bash
streamlit run app.py
```

---

## ☁️ Deployment

* Deployed using **Streamlit Community Cloud**
* Requires GitHub repository connection

---

## ⚠️ Important Notes

* Models must be generated before running app
* Run notebook once to create `Models/` folder
* Dataset: Amazon Alexa Reviews (Kaggle)

---

## 📚 Dataset

Amazon Alexa Reviews Dataset
Contains:

* Verified reviews
* Ratings
* Feedback labels

---

## 💡 Key Learnings

* Handling imbalanced datasets (SMOTE)
* Importance of TF-IDF over BoW for SVM
* Threshold tuning for better classification
* UI + ML integration using Streamlit
* Model explainability

---

## 🏁 Future Improvements

* Multi-class classification (Positive / Neutral / Negative)
* Deep learning (BERT / LSTM)
* Explainable AI (SHAP / LIME)
* Voice-based sentiment input


