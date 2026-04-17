import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("scam_dataset.csv")

# 🔥 FIX NAN ISSUE
df = df.dropna()
df["label"] = df["label"].astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)

# Vectorizer
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Accuracy
print("✅ Accuracy:", model.score(X_test_vec, y_test))

# Save model
pickle.dump((model, vectorizer), open("scam_model.pkl", "wb"))
print("✅ Model saved")