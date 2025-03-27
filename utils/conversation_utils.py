import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.util import ngrams
import spacy

lemmatizer = WordNetLemmatizer()

import spacy
import subprocess

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Ensure necessary NLTK resources are available
try:    
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    
try:    
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt_tab")
    
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
    
try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")

# Define stopwords for text analysis
custom_stopwords = ENGLISH_STOP_WORDS
sia = SentimentIntensityAnalyzer()

def extract_phrases(text_series, ngram_range=(2,3)):
    """Extracts common phrases (bigrams and trigrams) instead of single words."""
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words="english")
    X = vectorizer.fit_transform(text_series.dropna())
    phrases = Counter(vectorizer.get_feature_names_out())
    return phrases.most_common(5)  # Return top 5 phrases

def analyze_query(df, query, client):
    """Generate and execute Python code for data analysis and return the result."""
    if query:
        prompt = f"""
    The following is a dataset:
    {df.head()}

    The dataset contains the following columns: {df.columns.tolist()}

    User query: {query}

    Provide **only the Python code** to analyze the dataset based on the query.
    The code must be executable and should use pandas and NumPy if needed. 

    **Important Notes:**
    1. The dataset is stored in a variable called `df`.
    2. Ignore all the warnings
    3. Generate a final answer in dataframe for all kinds of queries
    4. Handle missing values (NaN) appropriately before processing.
    5. Ensure the output is stored in a variable called `result` (e.g., `result = df.describe()`).
    6. Do not include print statements, explanations, markdown formatting, or anything except valid Python code.
    7. If the query involves identifying the top issues in reviews, extract **key complaint phrases** using NLP techniques like **n-grams (bigrams/trigrams) or named entity recognition (NER)**.
    8. Remove stopwords and apply lemmatization before analysis.
    9. Please provide the following data in a proper dataframe format.
    """ 
        client_name = client.__class__.__name__.lower()

        if "groq" in client_name:  # Groq Client
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Provide only executable Python code for data analysis. Do not include any explanations or additional text."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile", 
                max_tokens=800
            )
        else:  # OpenAI Client
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Provide only executable Python code for data analysis. Do not include any explanations or additional text."},
                    {"role": "user", "content": prompt}
                ],
                model="gpt-4-turbo", 
                max_tokens=800
            )

        code_snippet = response.choices[0].message.content.strip()

        if not code_snippet:
            return "Error: Model returned an empty response."

        try:
            code_snippet = code_snippet.replace('```python', '').replace('```', '')
            print('Generated Code:', code_snippet)  # Debugging

            # Execution environment
            exec_globals = {
                'pd': pd,
                'np': np,
                'df': df,
                'Counter': Counter,
                'stop_words': custom_stopwords,
                'stopwords': custom_stopwords,
                'nltk': nltk,
                're': re,
                'word_tokenize': word_tokenize,
                'WordNetLemmatizer': WordNetLemmatizer,
                'lemmatizer': lemmatizer,
                'sia': sia,
                'SentimentIntensityAnalyzer': SentimentIntensityAnalyzer,
                'TfidfVectorizer': TfidfVectorizer,
                'CountVectorizer': CountVectorizer,
                'extract_phrases': extract_phrases,
                'nlp': nlp,
                'ngrams':ngrams
            }
            exec_locals = {}

            # Execute generated code
            try:
                exec(code_snippet, exec_globals, exec_locals)
            except Exception as exec_error:
                return f"Error executing generated code: {exec_error}"

            result = exec_locals.get("result", None)
            
            print("result:", result)  # Debugging

            # **Handle Different Response Formats**
            if isinstance(result, pd.DataFrame):
                # Ensure unique column names
                result.columns = [f"{col}_{i}" if list(result.columns).count(col) > 1 else col 
                                  for i, col in enumerate(result.columns)]
                return result  # Directly return DataFrame
            elif isinstance(result, list) and all(isinstance(i, tuple) and len(i) == 2 for i in result):
                return pd.DataFrame(result, columns=["Issue", "Frequency"])  # Convert list of tuples to DataFrame
            elif isinstance(result, (int, float, str, list, dict)):
                return result  # Return as-is for non-DataFrame responses
            else:
                return "Error: No valid output generated. The code may not have created a 'result' variable."

        except Exception as e:
            return f"Error executing the analysis: {e}"

    else:
        return "Please write a relevant query for data analysis."