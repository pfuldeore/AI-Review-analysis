import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
 
lemmatizer = WordNetLemmatizer()

# Define stopwords for text analysis
custom_stopwords = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)
sia = SentimentIntensityAnalyzer()
# Ensure necessary NLTK resources are available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")

def visualize_query(df, query, client, model="llama-3.3-70b-versatile"):
    """Generate and display a visualization based on the user query using Groq or OpenAI API."""
    if query:
        prompt = f"""
    The following is a dataset:
    {df.head()}

    The dataset contains the following columns: {df.columns.tolist()}

    User query: {query}

    Provide **only the Python altair code** to generate a plot based on the query. 
    The code should use `altair` and must be executable. Please add the relevant tooltip, chart title, and axis titles.
    Do not include any explanations, descriptions, or markdown formatting.
    The code should assume the dataset is stored in a variable called `df`.

    **Important Notes:**
    1. Handle missing values (NaN) in any text column by replacing them with empty strings before processing.
    2. Ensure the code is robust and does not fail if the dataset contains missing values.
    3. Do not include stopwords in the keyword analysis 
    4. Specify a type for encoding field 
    """
        client_name = client.__class__.__name__.lower()
        
        if "groq" in client_name: # Groq Client
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Provide only executable Python code for generating plots. Do not include any explanations or additional text."},
                    {"role": "user", "content": prompt}
                ],
                model=model, 
                max_tokens=500
            )
        else:  # OpenAI Client
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Provide only executable Python code for generating plots. Do not include any explanations or additional text."},
                    {"role": "user", "content": prompt}
                ],
                model="gpt-4-turbo", 
                max_tokens=500
            )

        code_snippet = response.choices[0].message.content.strip()

        try:
            if "chart.show()" in code_snippet or 'chart' in code_snippet:
                code_snippet = code_snippet.replace("chart.show()", "st.altair_chart(chart, use_container_width=True)")
                code_snippet = code_snippet.replace("chart.display()", "st.altair_chart(chart, use_container_width=True)")
                code_snippet = code_snippet.replace('```python', '').replace('```', '')

            print('Modified_Code: ', code_snippet)
            exec_globals = {
                'pd': pd,
                'np': np,
                'df': df,
                'alt':alt,
                'Counter': Counter,
                'ENGLISH_STOP_WORDS': ENGLISH_STOP_WORDS,
                'stop_words': custom_stopwords,
                'nltk': nltk,
                're': re,
                'word_tokenize': word_tokenize,
                'lemmatizer': lemmatizer,
                'sia': sia,
                'SentimentIntensityAnalyzer':SentimentIntensityAnalyzer
            }
            exec_locals = {}
            exec(code_snippet, exec_globals, exec_locals)

            if 'chart' in exec_locals:
                st.altair_chart(exec_locals['chart'], use_container_width=True)
            else:
                st.error("The generated code did not produce a chart. Please refine your query or try again.")

        except Exception as e:
            st.error(f"Error generating the plot: {e}")
            st.write("The generated code snippet was invalid. Please refine your query or try again.")

    else:
        st.warning("Please write a relevant query for visualization.")