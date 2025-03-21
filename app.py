import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from utils.api_utils import initialize_groq_client, initialize_openai_client
from utils.data_utils import load_and_preview_data
from utils.conversation_utils import analyze_query
from utils.sentiment_utils import filter_and_analyze_sentiment
from utils.visualization_utils import visualize_query
from utils.comparative_analysis_utils import comparative_analysis
import altair as alt

# Load environment variables from .env file
load_dotenv()

st.title("🤖 Shell Store Reviews Chatbot")

# Sidebar for API key selection
st.sidebar.header("Settings")
api_provider = st.sidebar.radio("Select LLM API Provider", ["Groq (Opensource)", "OpenAI (Paid)"])

if api_provider == "Groq (Opensource)":
    api_key = os.getenv("groq_API_KEY")
    client = initialize_groq_client(api_key) if api_key else None
elif api_provider == "OpenAI (Paid)":
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    client = initialize_openai_client(api_key) if api_key else None

if not api_key:
    st.sidebar.warning("Please provide a valid API key to proceed.")
else:
    # Section navigation
    section = st.sidebar.radio("Choose Section", ["Conversational AI", "Visualization", "Sentiment Analytics", "Comparative Analysis"])
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file containing customer reviews", type=["csv"])
    
    if uploaded_file is not None:
        df = load_and_preview_data(uploaded_file)
    
        if section == "Conversational AI":
            st.subheader("Ask a question about the dataset:")

            # Predefined queries
            predefined_queries = {
                    "Top 3 stores with highest average ratings": "Which are the top 3 stores with the highest average ratings?",
                    "Top 5 stores with most reviews": "Which are the top 5 stores with the most number of reviews?",
                    "Store with most 5-star ratings": "Which store has the most number of 5-star ratings?",
                    "Bottom 3 stores with lowest ratings": "Which are the bottom 3 stores with the lowest average ratings?",
                    "Bottom 5 stores with least reviews": "Which are the bottom 5 stores with the least number of reviews?",
                    "Store with most 1-star ratings": "Which store has the highest number of 1-star ratings?",
                    "Average rating of a store": "What is the average rating of [store_name]?",
                    "Most frequently reviewed store": "Which store has received the highest number of reviews?",
                    "Sentiment distribution of reviews": "What is the overall sentiment distribution of the reviews?",
                    "Common words in positive reviews": "What are the most common words in 5-star reviews?",
                    "Common words in negative reviews": "What are the most common words in 1-star reviews?",
                    "Review trends over time": "How have the number of reviews changed over time?",
                }

            # Layout for predefined query buttons
            selected_query = st.selectbox("Choose a predefined query:", [""] + list(predefined_queries.keys()))

            # Query input area
            query = st.text_area("Or type your own query:", value=predefined_queries.get(selected_query, ""))

            if st.button("Get Answer"):
                if query:
                    try:
                        response = analyze_query(df, query, client)  # Call function
                        
                        if isinstance(response, pd.DataFrame):
                            st.subheader("Analysis Result:")
                            st.dataframe(response)  # Display DataFrame
                        elif isinstance(response, str):  
                            st.subheader("Analysis Output:")
                            st.text_area("Result:", value=response, height=200)  # Display text output
                        else:
                            st.error("Unexpected response format. Please refine your query.")
                    except Exception as e:
                        st.error(f"Error executing analysis: {e}")
                else:
                    st.warning("Please enter a query.")

        elif section == "Visualization":
            st.subheader("Ask a question for a visualization")

            # Predefined visualization queries
            visualization_queries = {
                "Top 5 stores with highest average ratings": "visualize the top 5 stores with highest average ratings.",
                "Top 5 stores with most reviews": "visualize the top 5 stores with the most number of reviews.",
                "Store with most 5-star ratings": "visualize the store with the most 5-star ratings.",
                "Distribution of ratings": "visualize the distribution of ratings across all stores.",
                "Sentiment distribution of reviews": "visualize the sentiment distribution of reviews.",
                "Review trends over time": "visualize the number of reviews over time."
            }

            # Dropdown for predefined queries
            selected_query = st.selectbox("Choose a visualization query:", [""] + list(visualization_queries.keys()))

            # Chart type selection
            chart_types = {
                "Bar Chart": "bar chart",
                "Line Chart": "line chart",
                "Pie Chart": "pie chart",
                "Histogram": "histogram",
                "Scatter Plot": "scatter plot"
            }
            selected_chart = st.selectbox("Select chart type:", list(chart_types.keys()))

            # Query input area (pre-fills with predefined query if selected)
            query = st.text_area("Or type your own query:", 
                                value=f"Plot a {chart_types[selected_chart]} to {visualization_queries.get(selected_query, '')}" if selected_query else "")

            if st.button("Get Plot"):
                if query:
                    visualize_query(df, query, client)  # Call function
                else:
                    st.warning("Please enter a query.")


        
        elif section == "Sentiment Analytics":
            st.subheader("Filter Reviews")

            # Define necessary columns
            column_names = df.columns.tolist()
            station_col = "store_name"
            text_col = "text"
            rating_col = "Rating"
            date_col = "publishedAtDate"  

            # Convert Date column to datetime if not already
            df[date_col] = pd.to_datetime(df[date_col])

            # Define options for filtering
            unique_stations = ["All"] + df[station_col].unique().tolist()
            
            col1, col2 = st.columns(2)
            # Place the first selectbox in the first column
            with col1:
                 review_type_filter = st.selectbox(
                "Review On",
                ["All", "Staff", "Equipments", "Services", "Cleanliness", "Payment Experience", "Waiting Time", "Fuel Quality"],
            )
            # Place the second selectbox in the second column
            with col2:
                 station_filter = st.selectbox("Select store", unique_stations)
            
            # Filter the dataset and analyze sentiment
            filtered_df = filter_and_analyze_sentiment(df, station_filter, review_type_filter, station_col, text_col, rating_col)

            # Display filtered reviews with sentiment analysis
            st.dataframe(filtered_df[[station_col, text_col, "Sentiment", rating_col]])

            # ---- Monthly Trend of Average Ratings ----
            st.subheader("📈 Monthly Trend of Average Ratings")

            # Extract year-month and group by it
            filtered_df["YearMonth"] = filtered_df[date_col].dt.to_period("M").astype(str)  # Convert to Year-Month format
            monthly_trend = filtered_df.groupby("YearMonth")[rating_col].mean().reset_index()

            # Create Altair line chart
            line_chart = (
                alt.Chart(monthly_trend)
                .mark_line(point=True)  # Adds points to highlight data points
                .encode(
                    x=alt.X("YearMonth:T", title="Month"),
                    y=alt.Y(f"{rating_col}:Q", title="Average Rating"),
                    tooltip=[
                alt.Tooltip("YearMonth:T", title="Month", format="%b %Y"),  # Correct date format
                alt.Tooltip(f"{rating_col}:Q", title="Avg Rating"),
            ],)
                .properties(title=f"Monthly Trend of Average Ratings ({' '.join(station_filter.split()[:2])}, {review_type_filter})", width=700, height=400)
                .interactive()  # Allows zooming and panning
            )

            # Display chart in Streamlit
            st.altair_chart(line_chart, use_container_width=True)
            
        elif section == "Comparative Analysis":
            station_options = df["store_name"].unique().tolist()  # Select two stations
            # Create two columns
            col1, col2 = st.columns(2)
            # Place the first selectbox in the first column
            with col1:
                station1 = st.selectbox("Select First Station", station_options, index=0)
            # Place the second selectbox in the second column
            with col2:
                station2 = st.selectbox("Select Second Station", station_options, index=1)

            if st.button("Compare"):
                if station1 == station2:
                    st.warning("Please select two different stations for comparison.")
                else:
                    comparison_result = comparative_analysis(df, station1, station2, client)

                    # Display Metrics
                    st.markdown(f"### 📊 Comparison: **{' '.join(station1.split()[:2])} vs {' '.join(station2.split()[:2])}**")
                    st.dataframe(comparison_result)
