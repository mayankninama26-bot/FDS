import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# ✅ Removed a.csv
files = ["creditcard.csv", "test.csv", "Fraud_Data.csv"]

for f in files:
    file_path = os.path.join(script_dir, f)

    if not os.path.exists(file_path):
        print(f"❌ File not found: {f}")
        continue

    df = pd.read_csv(file_path)

    # Standardize target column
    if "Class" in df.columns:
        pass
    elif "is_fraud" in df.columns:
        df = df.rename(columns={"is_fraud": "Class"})
    else:
        print(f"⚠️ Skipping {f} (no target column)")
        continue

    # Keep numeric columns only
    df = df.select_dtypes(include=["int64", "float64"])

    # Remove missing labels
    df = df.dropna(subset=["Class"])

    # Skip bad dataset (only one class)
    if df["Class"].nunique() < 2:
        print(f"⚠️ Skipping {f} (only one class)")
        continue

    # Fill missing values
    df = df.fillna(0)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    if len(X.columns) == 0:
        print(f"⚠️ Skipping {f} (no usable features)")
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    model_name = f.replace(".csv", "")
    save_path = os.path.join(script_dir, f"{model_name}_model.pkl")

    pickle.dump((model, X.columns.tolist()), open(save_path, "wb"))

    print(f"✅ Model trained: {model_name}_model.pkl")

print("\n🔥 All models trained successfully!")