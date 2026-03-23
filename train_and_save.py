# ============================================================
#  train_and_save.py  –  20 Newsgroups Text Classifier
#  Run this ONCE to train and save the model artifacts.
#  Usage: python train_and_save.py
#  Output: model.pkl, vectorizer.pkl, labels.pkl
# ============================================================

import pickle
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. LOAD DATA
print("Loading 20 Newsgroups dataset...")
newsgroups   = fetch_20newsgroups(subset='all')
X_raw        = newsgroups.data
y            = newsgroups.target
target_names = newsgroups.target_names
print(f"Total documents : {len(X_raw)}")
print(f"Categories      : {len(target_names)}")

# 2. TF-IDF VECTORIZATION
print("\nVectorizing...")
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    sublinear_tf=True,
    max_features=70000,
    min_df=2
)
X = vectorizer.fit_transform(X_raw)

# 3. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. TRAIN ALL 3 MODELS, PICK BEST
models = [
    (LinearSVC(C=1.0, max_iter=2000),                            "LinearSVC"),
    (SGDClassifier(loss='hinge', penalty='l2', random_state=42), "SGDClassifier"),
    (MultinomialNB(alpha=0.01),                                   "MultinomialNB"),
]

best_model, best_model_name, best_acc = None, "", 0

print("\nTraining models...")
for model, name in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec  = recall_score(y_test, y_pred, average='weighted')
    print(f"  {name:20s}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
    if acc > best_acc:
        best_acc, best_model_name, best_model = acc, name, model

print(f"\nBest model: {best_model_name}  (Accuracy: {best_acc:.4f})")

# 5. SAVE ARTIFACTS
pickle.dump(best_model,   open("model.pkl",      "wb"))
pickle.dump(vectorizer,   open("vectorizer.pkl", "wb"))
pickle.dump(target_names, open("labels.pkl",     "wb"))

print("\n✅  Saved: model.pkl, vectorizer.pkl, labels.pkl")
print("    Now run:  streamlit run app.py")
