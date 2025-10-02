import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

print("Starting model training process...")

# 1. Load the Kaggle CSV data
# We use 'utf-8' encoding to handle different character types.
try:
    df = pd.read_csv('mental_health_faq.csv')
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

print("CSV file loaded successfully.")
print("Columns found:", df.columns.tolist())

# --- Important: Adapt to the CSV columns ---
# The columns in this CSV are 'Question_ID', 'Questions', 'Answers'.
# We will use 'Questions' for our input and 'Answers' for our output.
questions = df['Questions']
answers = df['Answers']

# 2. Create and train the TF-IDF Vectorizer
# This converts our text questions into meaningful numerical vectors.
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

print("Training TF-IDF Vectorizer...")
# The .fit_transform() method learns the vocabulary and transforms the data.
question_vectors = vectorizer.fit_transform(questions)
print("Vectorizer training complete.")

# 3. Save the trained artifacts
# We save the vectorizer (which knows the vocabulary) and the original data.
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
df.to_pickle('faq_df.pkl')

print("\nTraining complete!")
print("Model artifacts saved as 'tfidf_vectorizer.pkl' and 'faq_df.pkl'.")