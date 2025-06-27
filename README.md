# 📩 Spam vs Ham Message Classifier

This project uses a **Naive Bayes** machine learning model to classify messages as either **spam** or **ham** (non-spam) using natural language processing (NLP) techniques. The model is evaluated using metrics like accuracy, precision, recall, F1-score, and supported with visualizations such as a confusion matrix heatmap and word clouds.

## 📊 Features
- Text preprocessing using TF-IDF Vectorizer
- Model training using Multinomial Naive Bayes
- Performance evaluation using sklearn metrics
- Word cloud and clustering visualizations

## 🧪 Technologies Used
- Python
- Scikit-learn
- Pandas
- Matplotlib / Seaborn
- WordCloud
- TSNE / PCA

## 📂 Dataset
Used the classic **SMS Spam Collection Dataset** with labeled messages.

## 🚀 How to Run
```bash
pip install -r requirements.txt
python spam_classifier.py
