from flask import Flask, render_template, request, redirect, session, flash, url_for
import pandas as pd
import sqlite3
import pickle
import os
import random
import re
import numpy as np
import easyocr
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "fraud_ai_secret"

# ===============================
# EASY OCR (WORKS ON RENDER)
# ===============================
reader = easyocr.Reader(['en'], gpu=False)

# ===============================
# DATABASE
# ===============================
DB_PATH = os.environ.get("DB_PATH", "users.db")

def get_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fullname TEXT,
            email TEXT UNIQUE,
            mobile TEXT,
            password TEXT
        )
    """)
    db.commit()
    db.close()

init_db()


# ===============================
# LOAD MODELS
# ===============================
model_files = [
    "creditcard_model.pkl",
    "test_model.pkl",
    "Fraud_Data_model.pkl"
]

fraud_models = []

for f in model_files:
    if os.path.exists(f):
        try:
            model, features = pickle.load(open(f, "rb"))
            fraud_models.append((model, features))
        except:
            pass

# SCAM MODEL
try:
    scam_model, vectorizer = pickle.load(open("scam_model.pkl", "rb"))
except:
    scam_model = None
    vectorizer = None

# ===============================
# FOLDER
# ===============================
os.makedirs("uploads", exist_ok=True)

# ===============================
# MODEL DETECTION
# ===============================
def detect_best_model(df):
    best_model = None
    best_features = None
    max_match = 0

    for model, features in fraud_models:
        match = len(set(features).intersection(df.columns))
        if match > max_match:
            max_match = match
            best_model = model
            best_features = features

    return best_model, best_features

# ===============================
# LOGIN
# ===============================
@app.route("/", methods=["GET", "POST"])
def login():
    try:
        if request.method == "POST":
            email = request.form["email"]
            password = request.form["password"]

            db = get_db()
            user = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()

            if user and check_password_hash(user["password"], password):
                session["user"] = user["fullname"]
                session["email"] = user["email"]
                return redirect(url_for("dashboard"))
            else:
                flash("Invalid Username or Password")

        return render_template("login.html")
    except Exception as e:
        return f"Login Error: {e}"

# ===============================
# REGISTER
# ===============================

@app.route('/register', methods=['GET', 'POST'])
def register():
    try:
        if request.method == 'POST':
            fullname = request.form['fullname']
            email = request.form['email']
            mobile = request.form['mobile']
            password = request.form['password']
            confirm = request.form['confirm']

            if password != confirm:
                flash("Passwords do not match!")
                return render_template('register.html')

            if not re.fullmatch(r'[0-9]{10}', mobile):
                flash("Invalid mobile number!")
                return render_template('register.html')

            db = get_db()
            user = db.execute(
                "SELECT * FROM users WHERE email=? OR mobile=?",
                (email, mobile)
            ).fetchone()

            if user:
                flash("User already exists!")
                return render_template('register.html')

            hashed = generate_password_hash(password)

            db.execute(
                "INSERT INTO users (fullname, email, mobile, password) VALUES (?, ?, ?, ?)",
                (fullname, email, mobile, hashed)
            )
            db.commit()

            flash("Registration Successful!")
            return redirect(url_for('login'))

        return render_template('register.html')
    except Exception as e:
        return f"Register Error: {e}"
# ===============================
# FORGOT PASSWORD
# ===============================
@app.route("/forgot", methods=["GET", "POST"])
def forgot_password():
    try:
        if request.method == "POST":
            email = request.form.get("email")

            db = get_db()
            user = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()

            if user:
                otp = str(random.randint(1000, 9999))
                session["otp"] = otp
                session["reset_email"] = email

                print("OTP:", otp)

                flash("OTP sent (check console)")
                return redirect(url_for("verify_otp"))
            else:
                flash("Email not found")

        return render_template("forgot.html")
    except Exception as e:
        return f"Forgot Error: {e}"

# ===============================
# VERIFY OTP
# ===============================
@app.route("/verify_otp", methods=["GET", "POST"])
def verify_otp():
    try:
        if request.method == "POST":
            if request.form.get("otp") == session.get("otp"):
                return redirect(url_for("reset_password"))
            else:
                flash("Invalid OTP")

        return render_template("verify_otp.html")
    except Exception as e:
        return f"OTP Error: {e}"

# ===============================
# RESET PASSWORD
# ===============================
@app.route("/reset", methods=["GET", "POST"])
def reset_password():
    try:
        if request.method == "POST":
            password = request.form.get("password")
            confirm = request.form.get("confirm")

            if password != confirm:
                flash("Passwords do not match")
                return render_template("reset.html")

            email = session.get("reset_email")
            hashed = generate_password_hash(password)

            db = get_db()
            db.execute("UPDATE users SET password=? WHERE email=?", (hashed, email))
            db.commit()

            flash("Password reset successful!")
            return redirect(url_for("login"))

        return render_template("reset.html")
    except Exception as e:
        return f"Reset Error: {e}"
# ===============================
# DASHBOARD
# ===============================
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    safe = medium = high = 0
    results = None

    try:
        if request.method == "POST":
            csv_file = request.files.get("file")
            image_file = request.files.get("image")
            message = request.form.get("message")

            # ================= CSV =================
            if csv_file and csv_file.filename:
                path = os.path.join("uploads", csv_file.filename)
                csv_file.save(path)

                sample_df = pd.read_csv(path, nrows=200)
                model, features = detect_best_model(sample_df)

                if model is None:
                    flash("No suitable model")
                    return redirect(url_for("dashboard"))

                results_list = []

                for chunk in pd.read_csv(path, chunksize=10000):
                    chunk = chunk.reindex(columns=features, fill_value=0)
                    data = chunk.values

                    probs = model.predict_proba(data)[:, 1]

                    risk = np.where(probs > 0.59, "HIGH",
                            np.where(probs > 0.5, "MEDIUM", "LOW"))

                    chunk["Fraud_Score"] = probs
                    chunk["Risk_Level"] = risk

                    safe += np.sum(risk == "LOW")
                    medium += np.sum(risk == "MEDIUM")
                    high += np.sum(risk == "HIGH")

                    results_list.append(chunk.head(2))

                df_input = pd.concat(results_list, ignore_index=True)
                results = df_input.to_dict(orient="records")

            # ================= IMAGE =================
            elif image_file and image_file.filename:
                path = os.path.join("uploads", image_file.filename)
                image_file.save(path)

                try:
                    result = reader.readtext(path, detail=0)
                    text = " ".join(result)

                    if not text.strip():
                        text = "No text detected"

                except Exception as e:
                    text = f"OCR Error: {e}"

                prob = 0
                if scam_model and vectorizer and text:
                    prob = scam_model.predict_proba(vectorizer.transform([text]))[0][1]

                risk = "HIGH" if prob > 0.59 else "MEDIUM" if prob > 0.5 else "LOW"

                if risk == "LOW":
                    safe = 1
                elif risk == "MEDIUM":
                    medium = 1
                else:
                    high = 1

                results = [{
                    "Type": "Image",
                    "Text": text,
                    "Risk": risk,
                    "Score": round(prob * 100, 2)
                }]

            # ================= MESSAGE =================
            elif message:
                prob = 0
                if scam_model and vectorizer:
                    prob = scam_model.predict_proba(vectorizer.transform([message]))[0][1]

                risk = "HIGH" if prob > 0.59 else "MEDIUM" if prob > 0.5 else "LOW"

                if risk == "LOW":
                    safe = 1
                elif risk == "MEDIUM":
                    medium = 1
                else:
                    high = 1

                results = [{
                    "Type": "Message",
                    "Content": message,
                    "Risk": risk,
                    "Score": round(prob * 100, 2)
                }]

        return render_template("dashboard.html", safe=safe, medium=medium, high=high, results=results)

    except Exception as e:
        return f"Dashboard Error: {e}"

# ===============================
# LOGOUT
# ===============================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run(debug=False)
