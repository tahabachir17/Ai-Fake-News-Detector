import streamlit as st
import sys
import os
import pandas as pd
import time

# Add project root to path to allow imports from fake_news_detector
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fake_news_detector.models.predictor import FakeNewsPredictor
from fake_news_detector.data.preprocessor import TextPreprocessor

# Page Configuration
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main_header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        color: white;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .real {
        background-color: #4CAF50;
    }
    .fake {
        background-color: #F44336;
    }
    </style>
    """, unsafe_allow_html=True)

# Application Title
st.markdown('<div class="main_header">🕵️ AI Fake News Detector</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    """
    This application uses Machine Learning to detect fake news articles.
    
    **Available Models:**
    - Naive Bayes: Fast and effective baseline.
    - SVM: (Coming Soon)
    - Transformer: (Coming Soon)
    """
)
st.sidebar.title("Navigation")
st.sidebar.markdown("[Project Documentation](https://github.com/tahabachir17/Ai-Fake-News-Detector)")

# --- Main Logic ---

# Initialize Resources
@st.cache_resource
def load_resources():
    preprocessor = TextPreprocessor()
    # No extensive loading needed for preprocessor currently unless we preload heavy nltk data
    return preprocessor

preprocessor = load_resources()

# Input Section
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Analyze Article")
    article_text = st.text_area("Paste the news article text here:", height=300)

with col2:
    st.subheader("Configuration")
    model_choice = st.selectbox(
        "Select Model",
        ("Naive Bayes", "SVM", "Transformer")
    )
    
    analyze_button = st.button("Analyze Article", type="primary", use_container_width=True)

if analyze_button:
    if not article_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # Load Predictor (Lazy loading based on selection)
        # Note: We re-instantiate predictor to switch models if needed, 
        # but for caching purposes, we could cache the predictor itself.
        predictor = FakeNewsPredictor(model_name=model_choice.lower().replace(" ", "_"))
        
        if predictor.model is None and predictor.model_name == 'naive_bayes':
             st.error("Model not found! Has the model been trained yet? Please run training script.")
        elif predictor.model is None:
             st.info(f"{model_choice} model is not yet implemented. Please try Naive Bayes.")
        else:
            with st.spinner("Analyzing..."):
                # Simulating processing time for better UX
                time.sleep(0.5) 
                
                # Preprocess
                # Note: The predictor should ideally handle preprocessing or expect raw text if the pipeline includes it.
                # In train_nb.py, we saw the pipeline includes TfidfVectorizer but NOT TextPreprocessor?
                # Wait, train_nb.py did preprocessing manually THEN passed to TfidfVectorizer.
                # So we must preprocess here before passing to predictor.model.predict() if the saved model strictly expects clean text.
                # Checking train_nb.py: "grid_search.fit(X_train['text'], y_train)" where X_train['text'] was PREPROCESSED.
                # So YES, we must preprocess here.
                
                cleaned_text_series = preprocessor.transform([article_text])
                cleaned_text = cleaned_text_series.iloc[0]
                
                # Predict
                result = predictor.predict(cleaned_text)
                
            # Display Results
            if result['label'] == 'ERROR':
                st.error(result['message'])
            else:
                st.divider()
                r_col1, r_col2 = st.columns([1, 1])
                
                with r_col1:
                    # Confidence Metric
                    st.metric(label="Confidence Score", value=f"{result['score']:.2%}")
                    
                    # Result Box
                    label = result['label']
                    # Assuming standard labels, but adjusting for display
                    display_label = label.upper()
                    css_class = "real" if "REAL" in display_label else "fake"
                    
                    st.markdown(f'<div class="result-box {css_class}">{display_label}</div>', unsafe_allow_html=True)

                with r_col2:
                    st.subheader("Probability Distribution")
                    probs = result['probabilities']
                    prob_df = pd.DataFrame(list(probs.items()), columns=['Label', 'Probability'])
                    prob_df.set_index('Label', inplace=True)
                    st.bar_chart(prob_df)

