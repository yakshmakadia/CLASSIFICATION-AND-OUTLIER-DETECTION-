import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load and preprocess dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[['spamORham', 'Message']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['message'])
y = df['label']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Confusion Matrix Heatmap
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Word Clouds
spam_words = ' '.join(df[df['label'] == 1]['message'])
ham_words = ' '.join(df[df['label'] == 0]['message'])

WordCloud(width=600, height=400, background_color='black', colormap='Reds').generate(spam_words).to_image().show()
WordCloud(width=600, height=400, background_color='black', colormap='Greens').generate(ham_words).to_image().show()

# Optional: TSNE/PCA Clustering
use_tsne = True
reducer = TSNE(n_components=2, random_state=42) if use_tsne else PCA(n_components=2)
X_reduced = reducer.fit_transform(X.toarray())
colors = ['green' if label == 0 else 'red' for label in y]

plt.figure(figsize=(10, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=colors, alpha=0.5, s=10)
plt.title("Spam vs Ham Clustering")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Ham', markerfacecolor='green', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Spam', markerfacecolor='red', markersize=10)
])
plt.grid(True)
plt.show()
