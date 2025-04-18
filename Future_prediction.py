import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.utils import resample
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb
import random

# -------------------------------
# Global Feature Lists
# -------------------------------
numeric_features = [
    "Age",
    "Academic Performance (CGPA/Percentage)",
    "Risk-Taking Ability",
    "Financial Stability - self/family (1 is low income and 10 is high income)"
]

text_feature = "Preferred Subjects in Highschool/College"

categorical_features = [
    "Highest Education Level",
    "Preferred Work Environment",
    "Tech-Savviness"
]

target_col = "What would you like to become when you grow up"

# -------------------------------
# Data Augmentation
# -------------------------------
def augment_text(text):
    words = text.lower().split()
    if len(words) > 1:
        random.shuffle(words)
    if random.random() > 0.5:
        words.append(random.choice(["science", "mixed","commerce","arts"]))
    return ' '.join(words)

def augment_numeric(row):
    return row + np.random.normal(0, 0.1, size=row.shape)

def augment_data(df, n_augments=2):
    augmented_rows = []
    for _, row in df.iterrows():
        for _ in range(n_augments):
            new_row = row.copy()
            new_row[text_feature] = augment_text(row[text_feature])
            new_row[numeric_features] = augment_numeric(row[numeric_features])
            augmented_rows.append(new_row)
    return pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)

# -------------------------------
# Load, Clean, Augment, Preprocess Data
# -------------------------------
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df = df[numeric_features + categorical_features + [text_feature, target_col]].dropna()

    for col in numeric_features:
        df[col] = df[col].astype(str).str.extract(r"([\d.]+)").astype(float)
    df = df.dropna()

    df = augment_data(df)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target_col])

    tfidf = TfidfVectorizer(max_features=300, stop_words='english')
    X_text = tfidf.fit_transform(df[text_feature].astype(str))

    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[numeric_features])

    ohe_objects = []
    encoded_cats = []
    for col in categorical_features:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        transformed = ohe.fit_transform(df[[col]])
        ohe_objects.append(ohe)
        encoded_cats.append(transformed)
    X_cat = np.hstack(encoded_cats)

    X_combined = hstack([csr_matrix(X_num), csr_matrix(X_cat), X_text])

    return X_combined, y, tfidf, scaler, ohe_objects, label_encoder, df

# -------------------------------
# Balance Dataset
# -------------------------------
def balance_dataset(X, y):
    df_combined = pd.DataFrame(X.toarray())
    df_combined['target'] = y
    
    max_count = df_combined['target'].value_counts().max()
    balanced_df = pd.concat([
        resample(group, replace=True, n_samples=max_count, random_state=42)
        for _, group in df_combined.groupby('target')
    ])

    y_balanced = balanced_df['target'].values
    X_balanced = balanced_df.drop(columns=['target']).values
    return csr_matrix(X_balanced), y_balanced

# -------------------------------
# Train Ensemble Model
# -------------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    xgb_clf = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(np.unique(y)),
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )

    xgb_param_grid = {
        'max_depth': [6],
        'learning_rate': [0.1],
        'n_estimators': [350],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }

    xgb_grid = GridSearchCV(xgb_clf, xgb_param_grid, cv=3, verbose=1, n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    best_xgb = xgb_grid.best_estimator_

    svm_clf = SVC(kernel='rbf', C=5, probability=True)
    svm_clf.fit(X_train, y_train)

    ensemble = VotingClassifier(
        estimators=[("xgb", best_xgb), ("svm", svm_clf)],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    print(f"\n\U0001F3AF Final Ensemble Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return ensemble, xgb_grid, svm_clf, X_train, X_test, y_train, y_test

# -------------------------------
# Predict Future Career
# -------------------------------
def predict_future_career(model, user_input, tfidf, scaler, ohe_objects, label_encoder):
    age, education, subjects, performance, work_env, risk, tech, finance = user_input

    num = np.array([[age, performance, risk, finance]])
    X_num = scaler.transform(num)

    X_text = tfidf.transform([subjects])

    cats = [education, work_env, tech]
    encoded = []
    for i, val in enumerate(cats):
        transformed = ohe_objects[i].transform([[val]])
        encoded.append(transformed)
    X_cat = np.hstack(encoded)

    X_combined = hstack([csr_matrix(X_num), csr_matrix(X_cat), X_text])

    pred = model.predict(X_combined)
    career = label_encoder.inverse_transform(pred)[0]
    return career

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    X, y, tfidf, scaler, ohe_objects, label_encoder, df = load_and_preprocess_data("hack.csv")
    X_balanced, y_balanced = balance_dataset(X, y)
    model, xgb_model, svm_model, X_train, X_test, y_train, y_test = train_model(X_balanced, y_balanced)

    print("\n\U0001F393 Let's predict your future career!\n")
    try:
        age = int(input("Enter your Age: "))
        education_level = input("Enter your Highest Education Level (e.g., Bachelor's Degree): ")
        subjects = input("Preferred Subjects in Highschool/College (comma-separated): ")
        performance = float(input("Academic Performance (e.g., CGPA or %): "))
        work_env = input("Preferred Work Environment (e.g., Team-based, Independent): ")
        risk = int(input("Risk-Taking Ability (1 to 10): "))
        tech = input("Tech-Savviness (e.g., High, Medium, Low): ")
        finance = int(input("Financial Stability (1 to 10): "))

        user_input = (age, education_level, subjects, performance, work_env, risk, tech, finance)
        career = predict_future_career(model, user_input, tfidf, scaler, ohe_objects, label_encoder)

        print(f"\n\U0001F3AF Based on your input, your predicted career is: **{career}**")
    except Exception as e:
        print(f"\u26A0\uFE0F Error in input: {e}")