import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_top_keywords(reviews, top_n=10):
    """Extract top keywords from the reviews using TF-IDF."""
    if reviews.empty:
        return "No keywords available"

    vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform(reviews)
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]

    top_keywords = feature_array[tfidf_sorting][:top_n]
    return ", ".join(top_keywords)


def generate_keywords(df1, df2):
    """Generate structured comparison including keywords before calling LLM."""

    # Extract top keywords
    keywords1 = extract_top_keywords(df1["text"].dropna(), 10)
    keywords2 = extract_top_keywords(df2["text"].dropna(), 10)

    return keywords1, keywords2

def comparative_analysis(df, station1, station2, client, model="llama-3.3-70b-versatile"):
    """Perform comparative analysis with keyword extraction and improved LLM summary."""

    df1 = df[df["store_name"] == station1]
    df2 = df[df["store_name"] == station2]

    # Generate structured insights before calling LLM
    keywords1, keywords2 = generate_keywords(df1, df2)

    # LLM refines existing insights instead of analyzing all reviews
    prompt = f"""
    Customers frequently mention these topics in their reviews:
    - {station1}: {keywords1}
    - {station2}: {keywords2}

    Based on these keywords, refine this into a single sentence summary for each station. Do not add any preamble. 
    """
    client_name = client.__class__.__name__.lower()
    
    if "groq" in client_name:  # Groq Client
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=500
            )
    else:  # OpenAI Client
        response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4-turbo", 
                max_tokens=500
            )
    refined_summary = response.choices[0].message.content.strip()
    summary1, summary2 = refined_summary.split(".")[:2]  # Extract first two sentences

    # Construct DataFrame for comparison
    comparison_df = pd.DataFrame({
        "Metric": ["Total Reviews", "Average Rating", "Overall Rating", "Total 5⭐ Ratings", "Total 1⭐ Ratings", "Summary"],
        station1: [len(df1), round(df1["Rating"].mean(), 2), round(df1["Overall_Rating"].mean(), 2), len(df1[df1["Rating"] == 5]), len(df1[df1["Rating"] == 1]),  summary1],
        station2: [len(df2), round(df2["Rating"].mean(), 2), round(df2["Overall_Rating"].mean(), 2), len(df2[df2["Rating"] == 5]), len(df2[df2["Rating"] == 1]),  summary2]
    })

    return comparison_df




