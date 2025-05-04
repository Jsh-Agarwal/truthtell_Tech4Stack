import streamlit as st
import requests
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import plotly.graph_objects as go
from textblob import TextBlob

BASE_URL = "http://localhost:8002"  
BERT_URL = "http://localhost:8003" 

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        text = text.lower()  
        text = re.sub(r'<[^>]+>', '', text)  
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  
        text = re.sub(r'[^\w\s]', '', text)  
        text = re.sub(r'\d+', '', text)  
        tokens = word_tokenize(text)  
        tokens = [token for token in tokens if token not in self.stop_words]  
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]  
        return ' '.join(tokens)

def check_server_connection(url):
    try:
        requests.get(f"{url}/health", timeout=2)
        return True
    except:
        return False

def apply_custom_style():
    st.markdown("""
        <style>
        /* Main theme colors */
        :root {
            --primary-color: #4361ee;
            --secondary-color: #3bc9db;
            --accent-color: #f72585;
            --background-color: #0a1128;
            --card-color: rgba(255, 255, 255, 0.05);
        }

        /* Global styles */
        .stApp {
            background: linear-gradient(150deg, var(--background-color) 0%, #1a1b4b 100%);
            color: #ffffff;
        }

        /* Sidebar styling */
        .css-1d391kg {  /* Sidebar */
            background: linear-gradient(180deg, #1e2a78 0%, #1a1b4b 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .css-1d391kg .streamlit-button {
            background: var(--card-color);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            color: white;
            transition: all 0.3s ease;
        }

        /* Card containers */
        .custom-card {
            background: var(--card-color);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease;
        }
        .custom-card:hover {
            transform: translateY(-5px);
        }

        /* Headings */
        h1, h2, h3 {
            background: linear-gradient(120deg, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }

        /* Buttons and interactive elements */
        .stButton>button {
            background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(67, 97, 238, 0.3);
        }

        /* Text inputs and text areas */
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            background: var(--card-color);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            color: white;
            padding: 1rem;
        }

        /* Select boxes */
        .stSelectbox>div>div {
            background: var(--card-color);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            color: white;
        }

        /* Prediction boxes */
        .prediction-real {
            background: linear-gradient(45deg, #00b4d8, #90e0ef);
            color: white;
            border: none;
        }
        .prediction-fake {
            background: linear-gradient(45deg, #f72585, #b5179e);
            color: white;
            border: none;
        }

        /* Charts and visualizations */
        .js-plotly-plot {
            background: var(--card-color);
            border-radius: 15px;
            padding: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Loading spinner */
        .stSpinner > div {
            border-top-color: var(--primary-color) !important;
        }
        </style>
    """, unsafe_allow_html=True)

def homepage():
    st.markdown(
        """
        <style>
        /* Modern UI Theme */
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e7e7e7;
        }
        .welcome-text {
            font-size: 48px;
            font-weight: 800;
            background: linear-gradient(120deg, #00b4d8, #90e0ef);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 40px;
            padding: 20px;
        }
        .app-description {
            font-size: 20px;
            line-height: 1.8;
            text-align: center;
            color: #e7e7e7;
            margin: 20px auto;
            max-width: 800px;
        }
        .container {
            margin: 50px auto;
            padding: 40px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.markdown('<h1 class="welcome-text">Welcome to the TruthTell: Fake News Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="app-description">Harness the power of advanced machine learning to detect fake news with precision and accuracy.</p>', unsafe_allow_html=True)
    st.markdown('<p class="app-description">Select your preferred model, input your text, and get instant analysis with detailed visualizations.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def model_selection_page():
    st.markdown(
        """
        <style>
        .model-selection-container {
            padding: 40px;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            margin: 20px auto;
            max-width: 800px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .model-title {
            color: #00b4d8;
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 30px;
            text-align: center;
        }
        .stSelectbox > div {
            background: rgba(255, 255, 255, 0.1) !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: white !important;
        }
        .model-description {
            margin-top: 20px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            color: #e7e7e7;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        /* Custom dropdown styling */
        div[data-baseweb="select"] > div {
            background: rgba(255, 255, 255, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            transition: all 0.3s ease;
        }
        div[data-baseweb="select"] > div:hover {
            border-color: #00b4d8 !important;
            box-shadow: 0 0 15px rgba(0, 180, 216, 0.3);
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.markdown('<div class="model-selection-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="model-title">Choose Your Model</h2>', unsafe_allow_html=True)
    model = st.selectbox(
        "Select a classification model",
        ["Logistic Regression", "Naive Bayes", "SVM"]
    )
    if model:
        st.session_state.selected_model = model
        st.markdown(f'<div class="model-description">You have selected the <b>{model}</b> model.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def create_key_features_chart(features, scores):
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=features,
        y=scores,
        marker_color='#00b4d8',
        marker_line_color='#e7e7e7',
        marker_line_width=1.5,
        opacity=0.8
    ))
    
    fig.add_trace(go.Scatter(
        x=features,
        y=scores,
        mode='markers',
        marker=dict(
            color='#f72585',
            size=12,
            symbol='diamond',
            line=dict(color='#e7e7e7', width=1)
        ),
        name='Score'
    ))

    fig.update_layout(
        title={
            'text': "Key Features Impact",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, color='#e7e7e7')
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        font=dict(color='#e7e7e7'),
        showlegend=False,
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(size=12)
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def create_confidence_meter(confidence, prediction):
    """Create a clean confidence meter"""
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score", 'font': {'size': 24, 'color': '#e7e7e7'}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2fdf75" if prediction == 1 else "#ff6384"},
            'bgcolor': "rgba(0, 0, 0, 0)",
            'borderwidth': 2,
            'bordercolor': "#e7e7e7",
            'steps': [
                {'range': [0, 40], 'color': "rgba(255, 99, 132, 0.2)"},
                {'range': [40, 100], 'color': "rgba(47, 223, 117, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "#e7e7e7", 'width': 2},
                'thickness': 0.75,
                'value': 40
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=30, r=30, t=50, b=30),
        font={'color': "#e7e7e7"}
    )
    return fig

def create_text_statistics_chart(text):
    """Create detailed text statistics visualization"""
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    stats = {
        'Sentence Count': len(sentences),
        'Word Count': len(words),
        'Unique Words': len(set(words)),
        'Avg Words/Sentence': len(words) / len(sentences),
        'Complex Words': len([w for w in words if len(w) > 6])
    }
    
    fig = go.Figure([
        go.Bar(
            x=list(stats.keys()),
            y=list(stats.values()),
            marker_color='#00b4d8'
        )
    ])
    
    fig.update_layout(
        title="Text Statistics",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e7e7e7'},
        height=500,  
        xaxis={'tickangle': 45},
        margin=dict(l=50, r=50, t=80, b=100)  # Adjusted margins
    )
    return fig

def create_sentiment_analysis(text):
    """Create detailed sentiment analysis"""
    sentences = sent_tokenize(text)
    sentiments = [TextBlob(s).sentiment for s in sentences]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=[s.polarity for s in sentiments],
        name='Polarity',
        line={'color': '#00b4d8', 'width': 2},
        mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        y=[s.subjectivity for s in sentiments],
        name='Subjectivity',
        line={'color': '#f72585', 'width': 2},
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title="Sentiment Flow",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e7e7e7'},
        height=300,
        showlegend=True,
        legend={'bgcolor': 'rgba(255,255,255,0.1)'},
        xaxis_title="Sentence",
        yaxis_title="Score"
    )
    return fig

def create_prediction_banner(prediction, confidence):
    """Create a prominent prediction banner"""
    is_real = prediction == 1
    banner_color = "#2fdf75" if is_real else "#ff6384"
    banner_bg = "rgba(47, 223, 117, 0.1)" if is_real else "rgba(255, 99, 132, 0.1)"
    
    return f"""
        <div style="
            background: {banner_bg};
            border: 2px solid {banner_color};
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
            animation: pulse 2s infinite;
        ">
            <h1 style="
                color: {banner_color};
                font-size: 48px;
                margin: 0;
                font-weight: 800;
            ">
                {"REAL" if is_real else "FAKE"} NEWS
            </h1>
            <p style="
                color: #e7e7e7;
                font-size: 24px;
                margin: 10px 0;
            ">
                Confidence: {confidence:.1f}%
            </p>
        </div>
        <style>
        @keyframes pulse {{
            0% {{transform: scale(1);}}
            50% {{transform: scale(1.02);}}
            100% {{transform: scale(1);}}
        }}
        </style>
    """

def create_visualization_grid():
    """Create a clean CSS grid for visualizations"""
    return """
        <style>
        .visualization-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            padding: 20px 0;
        }
        .visualization-card {
            flex: 1 1 calc(50% - 10px);
            min-width: 400px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            height: auto;
            margin-bottom: 20px;
        }
        .chart-title {
            color: #00b4d8;
            font-size: 20px;
            font-weight: 600;
            text-align: center;
            margin-bottom: 15px;
            padding: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .chart-container {
            width: 100%;
            min-height: 400px;
            padding: 10px;
        }
        .js-plotly-plot {
            margin: 0 auto;
        }
        </style>
    """

def model_input_page():
    st.markdown(
        """
        <style>
        .input-container {
            padding: 40px;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            margin: 20px auto;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .model-header {
            color: #00b4d8;
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 30px;
            text-align: center;
        }
        .prediction-box {
            padding: 25px;
            border-radius: 15px;
            margin: 30px 0;
            text-align: center;
            font-size: 24px;
            font-weight: 600;
            backdrop-filter: blur(10px);
        }
        .prediction-real {
            background: rgba(47, 223, 117, 0.2);
            color: #2fdf75;
            border: 1px solid rgba(47, 223, 117, 0.4);
        }
        .prediction-fake {
            background: rgba(255, 99, 132, 0.2);
            color: #ff6384;
            border: 1px solid rgba(255, 99, 132, 0.4);
        }
        .visualization-container {
            background: rgba(255, 255, 255, 0.05);
            padding: 30px;
            border-radius: 15px;
            margin: 30px 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        /* Modern text area styling */
        .stTextArea textarea {
            background: #ffffff !important;  /* White background */
            border-radius: 15px !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            padding: 20px !important;
            font-size: 16px !important;
            color: #000000 !important;  /* Black text */
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        }
        .stTextArea textarea:focus {
            border-color: #00b4d8 !important;
            box-shadow: 0 0 20px rgba(0, 180, 216, 0.3) !important;
        }
        /* Label styling */
        .stTextArea label {
            color: #e7e7e7 !important;  /* Keep label text white */
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.markdown(f'<h2 class="model-header">{st.session_state.selected_model} Analysis</h2>', unsafe_allow_html=True)
    
    api_url = BERT_URL if st.session_state.selected_model == "BERT" else BASE_URL
    
    if not check_server_connection(api_url):
        st.error(f"⚠️ Unable to connect to the server at {api_url}")
        return

    st.write("Enter the text you want to classify:")
    user_input = st.text_area("Text input", height=200)

    if user_input:
        try:
            endpoint = ""
            if st.session_state.selected_model == "BERT":
                endpoint = f"{BERT_URL}/predict_bert"
            elif st.session_state.selected_model == "Logistic Regression":
                endpoint = f"{BASE_URL}/predict_logistic"
            elif st.session_state.selected_model == "Naive Bayes":
                endpoint = f"{BASE_URL}/predict_naive_bayes"
            elif st.session_state.selected_model == "SVM":
                endpoint = f"{BASE_URL}/predict_svm"

            with st.spinner('Processing...'):
                response = requests.post(endpoint, json={"text": user_input}, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result["predicted_label"]
                    message = result.get("message", "")
                    confidence = result.get("confidence", 0.0)
                    
                    st.markdown(create_visualization_grid(), unsafe_allow_html=True)
                    
                    st.markdown(
                        f"""
                        <div style='
                            background: {'rgba(47, 223, 117, 0.1)' if prediction == 1 else 'rgba(255, 99, 132, 0.1)'};
                            border: 2px solid {'#2fdf75' if prediction == 1 else '#ff6384'};
                            border-radius: 15px;
                            padding: 20px;
                            margin: 20px 0;
                            text-align: center;
                        '>
                            <h1 style='
                                color: {'#2fdf75' if prediction == 1 else '#ff6384'};
                                font-size: 36px;
                                margin: 0;
                            '>
                                {'REAL' if prediction == 1 else 'FAKE'} NEWS
                            </h1>
                            <p style='color: #e7e7e7; font-size: 20px; margin: 10px 0;'>
                                Confidence: {confidence:.1f}%
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    st.markdown('<div class="visualization-grid">', unsafe_allow_html=True)
                    
                    st.markdown('<div class="visualization-card">', unsafe_allow_html=True)
                    st.markdown('<div class="chart-title">Confidence Analysis</div>', unsafe_allow_html=True)
                    confidence_fig = create_confidence_meter(confidence, prediction)
                    st.plotly_chart(confidence_fig, use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="visualization-card">', unsafe_allow_html=True)
                    st.markdown('<div class="chart-title">Content Analysis</div>', unsafe_allow_html=True)
                    stats_fig = create_text_statistics_chart(user_input)
                    st.plotly_chart(stats_fig, use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="visualization-card">', unsafe_allow_html=True)
                    st.markdown('<div class="chart-title">Key Terms</div>', unsafe_allow_html=True)
                    wordcloud = WordCloud(
                        width=600,
                        height=400,
                        background_color='black',
                        colormap='viridis',
                        max_words=100
                    ).generate(user_input)
                    fig_wordcloud = plt.figure(figsize=(10, 6))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(fig_wordcloud)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="visualization-card">', unsafe_allow_html=True)
                    st.markdown('<div class="chart-title">Sentiment Flow</div>', unsafe_allow_html=True)
                    sentiment_fig = create_sentiment_analysis(user_input)
                    st.plotly_chart(sentiment_fig, use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                else:
                    st.error(f"Error: Server returned status code {response.status_code}")
                
        except Exception as e:
            st.error(f"Error processing request: {str(e)}")

def main():
    st.set_page_config(page_title="Truth Tell Fake News Classifier App", layout="wide")
    apply_custom_style() 
    
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e2a78 0%, #1a1b4b 100%);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: white;
            font-size: 18px;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio("Select a Page", ("Home", "Model Selection", "Model Input"))

    if page == "Home":
        homepage()
    elif page == "Model Selection":
        model_selection_page()
    elif page == "Model Input":
        if "selected_model" in st.session_state:
            model_input_page()
        else:
            st.error("Please select a model first!")

if __name__ == "__main__":
    main()