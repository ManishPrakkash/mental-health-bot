# main.py - TEMPORARY TEST VERSION 1
from fastapi import FastAPI
import sys

app = FastAPI(title="Simple Test API")

@app.get("/")
def read_root():
    # We can even add some diagnostic info
    py_version = sys.version
    return {
        "message": "Hello from Render! The simple API is working.",
        "python_version": py_version
    }