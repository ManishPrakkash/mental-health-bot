from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the FastAPI app
app = FastAPI(title="Mental Health FAQ Bot API")

# Define the structure for incoming requests
class UserQuery(BaseModel):
    question: str

# Load the trained model artifacts when the server starts
try:
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    faq_df = pd.read_pickle('faq_df.pkl')
    # Pre-compute the vectors for our knowledge base to make predictions faster
    question_vectors = vectorizer.transform(faq_df['Questions'])
except FileNotFoundError:
    print("Model files not found. Please run train_model.py first.")
    exit()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Mental Health FAQ Bot API. Post your question to /ask."}

@app.post("/ask")
def ask_question(query: UserQuery):
    user_question = query.question

    # Convert the user's question into a numerical vector
    user_question_vector = vectorizer.transform([user_question])

    # Calculate the similarity between the user's question and all questions in our dataset
    similarities = cosine_similarity(user_question_vector, question_vectors)

    # Find the index of the most similar question in our dataset
    most_similar_index = similarities.argmax()

    # Get the confidence score (the similarity value of the best match)
    confidence_score = similarities[0, most_similar_index]

    # Define a threshold to avoid irrelevant answers
    CONFIDENCE_THRESHOLD = 0.2

    if confidence_score >= CONFIDENCE_THRESHOLD:
        # If the match is good enough, return the corresponding answer
        answer = faq_df.iloc[most_similar_index]['Answers']
    else:
        # Otherwise, provide a fallback response
        answer = "I'm sorry, I don't have information on that topic. My knowledge is focused on mental health FAQs."

    return {
        "user_question": user_question,
        "predicted_answer": answer,
        "confidence_score": float(confidence_score) # Convert to standard float
    }