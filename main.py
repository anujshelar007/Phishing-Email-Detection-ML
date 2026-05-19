import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# =========================================
# CYBERSECURITY FEATURE FUNCTIONS
# =========================================

# Count URLs in email
def count_urls(text):
    urls = re.findall(r'http[s]?://\S+', text)
    return len(urls)


# Detect IP-based URLs
def has_ip_address(text):
    ip_pattern = r'http[s]?://(?:[0-9]{1,3}\.){3}[0-9]{1,3}'
    return 1 if re.search(ip_pattern, text) else 0


# Count suspicious phishing keywords
def suspicious_keywords(text):

    keywords = [
        'verify',
        'password',
        'bank',
        'urgent',
        'click',
        'login',
        'free',
        'account',
        'limited',
        'winner'
    ]

    text = text.lower()

    count = 0

    for word in keywords:
        if word in text:
            count += 1

    return count


# =========================================
# LOAD DATASET
# =========================================

print("\n==============================")
print(" PHISHING EMAIL DETECTION ML ")
print("==============================")

print("\nLoading Dataset...")

data = pd.read_csv("dataset.csv")

print("\nDataset Loaded Successfully!")

print("\nFirst 5 Rows of Dataset:\n")
print(data.head())


# =========================================
# CLEAN DATA
# =========================================

data = data.dropna()

print("\nMissing Values Removed!")


# =========================================
# CYBERSECURITY FEATURE EXTRACTION
# =========================================

print("\nExtracting Cybersecurity Features...")

data['url_count'] = data['Email Text'].apply(count_urls)

data['has_ip'] = data['Email Text'].apply(has_ip_address)

data['keyword_count'] = data['Email Text'].apply(suspicious_keywords)

print("\nCybersecurity Features Added Successfully!")

print("\nFeature Preview:\n")

print(
    data[
        [
            'url_count',
            'has_ip',
            'keyword_count'
        ]
    ].head()
)


# =========================================
# INPUT AND OUTPUT
# =========================================

X = data['Email Text']

y = data['Email Type']


# Convert labels into numbers
y = y.map({
    'Phishing Email': 1,
    'Safe Email': 0
})

print("\nLabels Converted:")
print("Phishing Email = 1")
print("Safe Email = 0")


# =========================================
# TEXT FEATURE EXTRACTION
# =========================================

print("\nConverting Email Text into Numerical Data...")

vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

print("Text Vectorization Completed!")


# =========================================
# SPLIT DATASET
# =========================================

print("\nSplitting Dataset into Training and Testing Data...")

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Size:", X_train.shape[0])
print("Testing Data Size:", X_test.shape[0])


# =========================================
# TRAIN MACHINE LEARNING MODEL
# =========================================

print("\nTraining Machine Learning Model...")

model = LogisticRegression()

model.fit(X_train, y_train)

print("Model Training Completed!")


# =========================================
# PREDICTIONS
# =========================================

print("\nTesting Model...")

predictions = model.predict(X_test)

print("Predictions Completed!")


# =========================================
# ACCURACY
# =========================================

accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print(" MODEL PERFORMANCE ")
print("==============================")

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nMeaning:")
print("Out of 100 emails, the model predicts")
print("about 96 emails correctly.")


# =========================================
# CONFUSION MATRIX
# =========================================

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

print("\nConfusion Matrix Explanation:")

print(f"""
1. Safe Emails Correctly Predicted as Safe:
   {cm[0][0]}

2. Safe Emails Incorrectly Predicted as Phishing:
   {cm[0][1]}

3. Phishing Emails Incorrectly Predicted as Safe:
   {cm[1][0]}

4. Phishing Emails Correctly Predicted as Phishing:
   {cm[1][1]}
""")


# =========================================
# VISUALIZE CONFUSION MATRIX
# =========================================

print("Displaying Confusion Matrix Graph...")

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Safe', 'Phishing'],
    yticklabels=['Safe', 'Phishing']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Phishing Email Detection Confusion Matrix")

plt.show()


# =========================================
# CUSTOM EMAIL TEST
# =========================================

print("\n==============================")
print(" CUSTOM EMAIL TEST ")
print("==============================")

sample_email = [
    "Urgent! Verify your bank account immediately by clicking this link http://192.168.1.1/login"
]

print("\nTesting Email:\n")
print(sample_email[0])

print("\nCybersecurity Analysis:")

print("URL Count:", count_urls(sample_email[0]))

print("Contains IP Address URL:",
      has_ip_address(sample_email[0]))

print("Suspicious Keyword Count:",
      suspicious_keywords(sample_email[0]))


sample_vector = vectorizer.transform(sample_email)

result = model.predict(sample_vector)

print("\nFinal Prediction:")

if result[0] == 1:
    print("⚠️ PHISHING EMAIL DETECTED")
else:
    print("✅ SAFE EMAIL")


print("\nProject Execution Completed Successfully!")