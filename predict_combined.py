import pickle
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))

# ✅ Only valid models
model_files = [
    "creditcard_model.pkl",
    "test_model.pkl",
    "Fraud_Data_model.pkl"
]

models = []

# Load models
for f in model_files:
    path = os.path.join(script_dir, f)

    if os.path.exists(path):
        model, features = pickle.load(open(path, "rb"))
        models.append((model, features))
        print(f"✅ Loaded {f}")
    else:
        print(f"❌ Missing model file: {f}")


def predict_all(input_data):
    predictions = []
    confidences = []

    for model, features in models:
        # ✅ Create DataFrame with correct feature names
        df_input = pd.DataFrame([input_data])

        # Align columns with model
        df_input = df_input.reindex(columns=features, fill_value=0)

        pred = int(model.predict(df_input)[0])

        # ✅ SAFE probability handling
        probs = model.predict_proba(df_input)[0]

        if len(probs) > 1:
            prob = probs[1]
        else:
            prob = probs[0]

        predictions.append(pred)
        confidences.append(prob)

    # Majority voting
    final = max(set(predictions), key=predictions.count)

    # Average confidence
    avg_conf = sum(confidences) / len(confidences)

    return final, predictions, avg_conf


# 🔥 Test run
if __name__ == "__main__":
    sample_input = {
        "V1": 0.2,
        "V2": -1.5,
        "Amount": 300,
        "Time": 10000
    }

    final, preds, conf = predict_all(sample_input)

    print("\nAll model predictions:", preds)
    print("Final Prediction:", final)
    print("Confidence:", round(conf * 100, 2), "%")