# train_model.py
import re, string, pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Demo data if you don't have spam.csv
data = [
    ("WIN a free ticket now!", "spam"),
    ("Please review the project report.", "ham"),
    ("Limited time offer — buy now!", "spam"),
    ("Can we meet Monday at 10?", "ham")
]
df = pd.DataFrame(data, columns=["text","label"])
df['label'] = df['label'].map(lambda x: 1 if x == 'spam' else 0)
df['text'] = df['text'].apply(clean_text)

# If you have spam.csv (optional), uncomment and adjust columns:
# df = pd.read_csv("spam.csv", encoding='latin-1')
# df = df[['v2','v1']]  # example if v2=text, v1=label
# df.columns = ['text','label']
# df['label'] = df['label'].map(lambda x: 1 if x.strip().lower() in ['spam','1'] else 0)
# df['text'] = df['text'].apply(clean_text)

X_text = df['text'].values
y = df['label'].values

vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2))
X = vectorizer.fit_transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"Validation accuracy: {acc:.3f}")

pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("Saved spam_model.pkl and vectorizer.pkl")
