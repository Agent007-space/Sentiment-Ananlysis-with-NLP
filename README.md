# Sentiment-Ananlysis-with-NLP
import nltk
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Sample dataset
data = [
    ("I love this product, it's amazing!", 1),
    ("This is the best thing I ever bought.", 1),
    ("I am very happy with the quality.", 1),
    ("Absolutely fantastic service!", 1),
    ("I hate it, total waste of money.", 0),
    ("Worst purchase I have made.", 0),
    ("Very disappointing experience.", 0),
    ("Not good at all, I want a refund.", 0)
]

# Split into texts and labels
texts, labels = zip(*data)

# Preprocessing function
def preprocess(text):
    # Lowercase, remove punctuation, remove stopwords
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return ' '.join(tokens)

# Apply preprocessing
texts = [preprocess(text) for text in texts]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25, random_state=42)

# Vectorization
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Function for real-time prediction
def predict_sentiment(text):
    processed = preprocess(text)
    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)[0]
    return "Positive" if prediction == 1 else "Negative"

# Test
print(predict_sentiment("I really love this!"))
print(predict_sentiment("This is terrible."))
