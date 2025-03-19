import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from textblob import TextBlob
from sklearn.decomposition import PCA

# Set page configuration
st.set_page_config(
    page_title="Laptop Reviews Analysis",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton button {
        background-color: #4F8BF9;
        color: white;
    }
    .stTextInput input, .stNumberInput input {
        background-color: #262730;
        color: white;
    }
    .stSelectbox, .stMultiselect {
        background-color: #262730;
    }
    </style>
""", unsafe_allow_html=True)

# Load the saved model and vectorizer
@st.cache_resource
def load_model():
    try:
        with open("models/rating_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        with open("models/vectorizer.pkl", "rb") as vec_file:
            vectorizer = pickle.load(vec_file)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please make sure the model files are in the 'models/' directory.")
        return None, None

# Load the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset/laptops_dataset_final_600.csv")
        
        # Clean up the data
        for col in ['no_ratings', 'no_reviews']:
            df[col] = df[col].astype(str).str.replace(',', '').replace('nan', np.nan).astype(float)
        
        # Fill NA values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        text_cols = df.select_dtypes(include=[object]).columns
        for col in text_cols:
            df[col].fillna('', inplace=True)
            
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please make sure the dataset file is in the 'dataset/' directory.")
        return None

# Main function
def main():
    st.title("Laptop Reviews Analysis Dashboard")
    
    # Load model and data
    model, vectorizer = load_model()
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Overview", "Rating Distribution", "Correlation Analysis", "Sentiment Analysis", "Review Prediction"])
    
    if page == "Data Overview":
        st.header("Dataset Overview")
        
        # Display basic dataset information
        st.subheader("Dataset Sample")
        st.dataframe(df.head())
        
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Total Records: {df.shape[0]}")
            st.write(f"Total Features: {df.shape[1]}")
        with col2:
            st.write(f"Numeric Features: {df.select_dtypes(include=[np.number]).shape[1]}")
            st.write(f"Categorical Features: {df.select_dtypes(include=[object]).shape[1]}")
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())
        
        # Display column names for debugging
        st.subheader("Available Columns")
        st.write(df.columns.tolist())
        
    elif page == "Rating Distribution":
        st.header("Rating Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['overall_rating'], kde=True, bins=10, color='skyblue', ax=ax)
            ax.set_title('Distribution of Overall Rating')
            ax.set_xlabel('Overall Rating')
            ax.set_facecolor('#0E1117')
            fig.set_facecolor('#0E1117')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['rating'], kde=True, bins=10, color='salmon', ax=ax)
            ax.set_title('Distribution of Rating')
            ax.set_xlabel('Rating')
            ax.set_facecolor('#0E1117')
            fig.set_facecolor('#0E1117')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            st.pyplot(fig)
        
        # Count plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='rating', data=df, palette='viridis', ax=ax)
        ax.set_title('Count Plot for Rating')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        ax.set_facecolor('#0E1117')
        fig.set_facecolor('#0E1117')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for text in ax.texts:
            text.set_color('white')
        st.pyplot(fig)
        
    elif page == "Correlation Analysis":
        st.header("Correlation Analysis")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] >= 4:
            fig, ax = plt.subplots(figsize=(12, 10))
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            ax.set_title('Correlation Heatmap')
            ax.set_facecolor('#0E1117')
            fig.set_facecolor('#0E1117')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            st.pyplot(fig)
        else:
            st.warning('Not enough numeric columns for a correlation heatmap')
        
    elif page == "Sentiment Analysis":
        st.header("Sentiment Analysis")
        
        # Calculate sentiment scores if not already done
        if 'sentiment' not in df.columns:
            with st.spinner("Calculating sentiment scores..."):
                df['sentiment'] = df['review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='sentiment', y='rating', data=df, ax=ax)
        ax.set_title("Sentiment Score vs Rating")
        ax.set_xlabel("Sentiment Score")
        ax.set_ylabel("Rating")
        ax.set_facecolor('#0E1117')
        fig.set_facecolor('#0E1117')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        st.pyplot(fig)
        
        # Display top reviewed laptops by sentiment
        st.subheader("Top Positive Reviews")
        top_positive = df.sort_values('sentiment', ascending=False).head(5)
        for i, row in enumerate(top_positive.iterrows(), 1):
            row = row[1]
            # Use a generic identifier since there's no laptop_name column
            st.markdown(f"**{i}. Review #{row.name}** - Rating: {row['rating']}, Sentiment: {row['sentiment']:.2f}")
            st.write(f"Review: {row['review'][:200]}...")
        
        st.subheader("Top Negative Reviews")
        top_negative = df.sort_values('sentiment').head(5)
        for i, row in enumerate(top_negative.iterrows(), 1):
            row = row[1]
            # Use a generic identifier since there's no laptop_name column
            st.markdown(f"**{i}. Review #{row.name}** - Rating: {row['rating']}, Sentiment: {row['sentiment']:.2f}")
            st.write(f"Review: {row['review'][:200]}...")
        
    elif page == "Review Prediction":
        st.header("Review Rating Prediction")
        
        if model is None or vectorizer is None:
            st.warning("Model not loaded. Please check the 'models/' directory.")
            st.stop()
        
        user_review = st.text_area("Enter a laptop review:", height=150)
        
        if st.button("Predict Rating"):
            if user_review:
                # Preprocess and predict
                review_tfidf = vectorizer.transform([user_review])
                prediction = model.predict(review_tfidf)[0]
                sentiment = TextBlob(user_review).sentiment.polarity
                
                # Display prediction
                st.success(f"Predicted Rating: {prediction}")
                
                # Display sentiment
                st.subheader("Sentiment Analysis")
                sentiment_text = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
                st.write(f"Sentiment: {sentiment_text} ({sentiment:.2f})")
                
                # Display confidence (if available)
                try:
                    proba = model.predict_proba(review_tfidf)
                    st.subheader("Prediction Confidence")
                    for i, rating in enumerate(model.classes_):
                        st.write(f"Rating {rating}: {proba[0][i]:.2f}")
                except:
                    pass
            else:
                st.error("Please enter a review.")

if __name__ == "__main__":
    main()