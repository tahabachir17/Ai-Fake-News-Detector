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
    .input-info {
        padding: 8px 12px;
        border-radius: 5px;
        background-color: #E3F2FD;
        color: #1565C0;
        font-size: 0.9rem;
        margin-top: 10px;
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
    - SVM: Linear SVM with TF-IDF
    - Transformer: (Coming Soon)
    
    **Input Options:**
    - Paste article text directly
    - Provide a URL to a news article
    """
)
st.sidebar.title("Navigation")
st.sidebar.markdown("[Project Documentation](https://github.com/tahabachir17/Ai-Fake-News-Detector)")

# --- Main Logic ---

# Initialize Resources
@st.cache_resource
def load_predictor(model_name):
    return FakeNewsPredictor(model_name=model_name)

# Input Section
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Analyze Article")
    
    # Input type selection
    input_type = st.radio(
        "Choose input type:",
        ("Text", "URL"),
        horizontal=True
    )
    
    if input_type == "Text":
        user_input = st.text_area(
            "Paste the news article text here:",
            height=300,
            placeholder="Paste the full article text here..."
        )
    else:
        user_input = st.text_input(
            "Enter the article URL:",
            placeholder="https://www.example.com/news-article"
        )

with col2:
    st.subheader("Configuration")
    model_choice = st.selectbox(
        "Select Model",
        ("Naive Bayes", "SVM", "Transformer")
    )
    
    analyze_button = st.button(
        "🔍 Analyze Article",
        type="primary",
        use_container_width=True
    )

if analyze_button:
    if not user_input or not user_input.strip():
        st.warning("Please enter some text or a URL to analyze.")
    else:
        # Load Predictor (cached per model name)
        model_key = model_choice.lower().replace(" ", "_")
        predictor = load_predictor(model_name=model_key)
        
        if predictor.model is None and predictor.model_name == 'naive_bayes':
            st.error("Model not found! Has the model been trained yet? Please run the training script.")
        elif predictor.model is None:
            st.info(f"{model_choice} model is not yet implemented. Please try Naive Bayes.")
        else:
            with st.spinner("Analyzing..."):
                time.sleep(0.3)
                
                # Predict — preprocessor handles URL detection + text cleaning internally
                result = predictor.predict(user_input.strip())
            
            # Display Results
            if result['label'] == 'ERROR':
                st.error(f"❌ Error: {result.get('message', 'Unknown error')}")
            else:
                st.divider()
                
                # Show input type info
                detected = result.get('input_type', 'text')
                if detected == 'url':
                    st.markdown(
                        '<div class="input-info">📡 Article text was extracted from the provided URL</div>',
                        unsafe_allow_html=True
                    )
                
                r_col1, r_col2 = st.columns([1, 1])
                
                with r_col1:
                    # Confidence Metric
                    st.metric(
                        label="Confidence Score",
                        value=f"{result['score']:.2%}"
                    )
                    
                    # Result Box
                    label = result['label']
                    display_label = label.upper()
                    css_class = "real" if "REAL" in display_label or display_label == "0" else "fake"
                    display_text = "✅ REAL NEWS" if css_class == "real" else "🚨 FAKE NEWS"

                    st.markdown(
                        f'<div class="result-box {css_class}">{display_text}</div>',
                        unsafe_allow_html=True
                    )

                with r_col2:
                    st.subheader("Probability Distribution")
                    probs = result['probabilities']
                    prob_df = pd.DataFrame(
                        list(probs.items()),
                        columns=['Label', 'Probability']
                    )
                    prob_df.set_index('Label', inplace=True)
                    st.bar_chart(prob_df)
