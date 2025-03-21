import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Define related keywords for each review type
REVIEW_TYPE_KEYWORDS = {
    "Staff": ["staff", "employee", "worker", "attendant", "friendly", "rude", "helpful", "service"],
    "Equipments": ["equipment", "machine", "pump", "dispenser", "functioning", "broken", "modern", "outdated"],
    "Services": ["service", "assistance", "support", "quality", "efficient", "slow", "professional", "care"],
    "Cleanliness": ["clean", "dirty", "hygiene", "sanitation", "messy", "tidy", "neat", "filthy"],
    "Payment Experience": ["payment", "card", "cash", "digital", "transaction", "easy", "difficult", "smooth"],
    "Waiting Time": ["wait", "time", "queue", "long", "short", "delay", "fast", "slow"],
    "Fuel Quality": ["fuel", "petrol", "diesel", "quality", "pure", "adulterated", "efficient", "performance"],
}

def get_sentiment_label_vader(review, rating=None):
    """Analyze sentiment using VADER and incorporate rating if available."""
    if pd.notna(review):
        sentiment_scores = sia.polarity_scores(review)
        compound_score = sentiment_scores["compound"]  # VADER's sentiment score (-1 to 1)

        # If rating is given, use it to adjust sentiment classification
        if rating is not None:
            if rating <= 2 and compound_score > 0:
                return "😐 Neutral"  # 1-star with positive text = possible issue
            elif rating >= 4 and compound_score < 0:
                return "😐 Neutral"  # 5-star with negative text = possible issue
        
        # Classify sentiment
        if compound_score > 0.2:
            return "😀 Positive"
        elif compound_score < -0.2:
            return "😞 Negative"
        else:
            return "😐 Neutral"
    
    return "😐 Neutral"

def filter_by_semantic_similarity(df, review_type_filter, threshold=0.2):
    """Filter reviews based on semantic similarity to the review type."""
    if review_type_filter == "All":
        return df

    # Get related keywords for the review type
    keywords = REVIEW_TYPE_KEYWORDS.get(review_type_filter, [review_type_filter])

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["text"].fillna(""))

    # Calculate similarity between reviews and keywords
    keyword_tfidf = vectorizer.transform([" ".join(keywords)])
    similarities = cosine_similarity(tfidf_matrix, keyword_tfidf).flatten()

    # Filter reviews with similarity above the threshold
    filtered_df = df[similarities >= threshold]
    return filtered_df

def filter_and_analyze_sentiment(df, station_filter, review_type_filter, station_col, text_col, rating_col):
    """Filter the dataset and analyze sentiment."""
    filtered_df = df.copy()
    
    if station_filter != "All":
        filtered_df = filtered_df[filtered_df[station_col] == station_filter]
    if review_type_filter != "All":
        filtered_df = filter_by_semantic_similarity(filtered_df, review_type_filter)

    # Apply improved sentiment function
    filtered_df["Sentiment"] = filtered_df.apply(lambda row: get_sentiment_label_vader(row[text_col], row.get(rating_col, None)), axis=1)
    
    return filtered_df