# main.py - TEMPORARY TEST VERSION 2
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

print("--- Application starting up ---")

app = FastAPI(title="Mental Health FAQ Bot API - Debug Mode")

try:
    print("Checkpoint 1: Attempting to load tfidf_vectorizer.pkl...")
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("Checkpoint 2: Vectorizer loaded successfully!")

    print("Checkpoint 3: Attempting to load faq_df.pkl...")
    faq_df = pd.read_pickle('faq_df.pkl')
    print("Checkpoint 4: DataFrame loaded successfully!")

    print("Checkpoint 5: Transforming questions with vectorizer...")
    question_vectors = vectorizer.transform(faq_df['Questions'])
    print("Checkpoint 6: All model artifacts loaded and ready!")

except Exception as e:
    print(f"AN ERROR OCCURRED DURING STARTUP: {e}")
    # We will re-raise the exception to ensure the app still fails if there's an error
    raise

class UserQuery(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: UserQuery):
    # This part of the code is not relevant to the startup crash
    # But we leave it here for when it works.
    user_question_vector = vectorizer.transform([query.question])
    similarities = cosine_similarity(user_question_vector, question_vectors)
    most_similar_index = similarities.argmax()
    confidence_score = similarities[0, most_similar_index]

    if confidence_score >= 0.2:
        answer = faq_df.iloc[most_similar_index]['Answers']
    else:
        answer = "I'm sorry, I don't have information on that topic."

    return {
        "predicted_answer": answer,
        "confidence_score": float(confidence_score)
    }