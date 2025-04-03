from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import string
from textblob import TextBlob
import logging
from nltk.tag import pos_tag
from nltk.metrics import edit_distance
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

try:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")

logistic_model_path = r"C:\Users\Lenovo\OneDrive\Desktop\Truthtell_Tech4Stack\logistic_model.pkl"
naive_bayes_model_path = r"C:\Users\Lenovo\OneDrive\Desktop\Truthtell_Tech4Stack\naive_bayes_model.pkl"
svm_model_path = r"C:\Users\Lenovo\OneDrive\Desktop\Truthtell_Tech4Stack\svm_model.pkl"

# Verify model loading
try:
    logger.info("Loading models...")
    logistic_model = joblib.load(logistic_model_path)
    naive_bayes_model = joblib.load(naive_bayes_model_path)
    svm_model = joblib.load(svm_model_path)
    logger.info("All models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise


class TextPreprocessor:
    def __init__(self):
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except LookupError as e:
            logger.error(f"NLTK resource not found: {e}")
            # Fallback to empty set if stopwords unavailable
            self.stop_words = set()
        except Exception as e:
            logger.error(f"Error initializing TextPreprocessor: {e}")
            self.stop_words = set()
    
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

def check_news_patterns(text):
    """Enhanced check for news article patterns with scoring"""
    try:
        score = 0
        reasons = []

        # 1. Check for date patterns with higher weight
        date_patterns = [
            r'\w+ \d{1,2},? \d{4}',  # April 3, 2025
            r'\d{1,2}/\d{1,2}/\d{4}',  # 4/3/2025
            r'FY\d{2,4}',  # FY24, FY2024
            r'\d{4}'  # Just year
        ]
        if any(re.search(pattern, text) for pattern in date_patterns):
            score += 2
            reasons.append("Contains date")

        # 2. Check for monetary/statistical patterns
        money_patterns = [
            r'\$\d+(?:\.\d+)?(?:\s*(?:billion|million|thousand))?',
            r'\d+(?:\.\d+)?%',
            r'\d+(?:\.\d+)?(?:\s*(?:billion|million|thousand))'
        ]
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in money_patterns):
            score += 2
            reasons.append("Contains statistics")

        # 3. Check for news-specific phrases
        news_phrases = [
            'reported', 'according to', 'said', 'announced', 'stated',
            'confirmed', 'officials', 'sources', 'spokesperson'
        ]
        if any(phrase in text.lower() for phrase in news_phrases):
            score += 2
            reasons.append("Contains attributions")

        # 4. Check for proper nouns and organizations
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        proper_nouns = sum(1 for word, tag in pos_tags if tag.startswith('NNP'))
        if proper_nouns >= 3:
            score += 2
            reasons.append("Contains proper nouns")

        # 5. Check for quotes
        if '"' in text or '"' in text or '"' in text:
            score += 1
            reasons.append("Contains quotes")

        # 6. Check paragraph structure
        paragraphs = text.split('\n\n')
        if len(paragraphs) >= 2:
            score += 1
            reasons.append("Multiple paragraphs")

        # Return detailed result
        return {
            'is_news': score >= 4,  # Lowered threshold
            'score': score,
            'reasons': reasons
        }
    except Exception as e:
        logger.error(f"Error in news pattern check: {e}")
        return {'is_news': False, 'score': 0, 'reasons': ['Error in analysis']}

def analyze_content_quality(text):
    """Enhanced analysis focused on news content"""
    try:
        # Basic text cleanup
        words = text.split()
        sentences = sent_tokenize(text)
        
        # Minimum length requirement
        if len(words) < 30:  # Lowered minimum length
            return False, "Text too short for a news article"
        
        # Check news patterns
        news_check = check_news_patterns(text)
        if not news_check['is_news']:
            return False, f"Missing news characteristics. Score: {news_check['score']}/8"
        
        # Basic structural checks
        if len(sentences) < 2:
            return False, "Too few sentences"
            
        # More lenient word repetition check
        word_freq = Counter(words)
        if any(freq > len(words) * 0.2 for freq in word_freq.values()):
            return False, "Excessive word repetition"

        return True, f"Valid news content. Features: {', '.join(news_check['reasons'])}"
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return False, f"Analysis error: {str(e)}"

def predict_with_validation(model, text, preprocessor):
    """Modified prediction with more lenient validation"""
    try:
        if not isinstance(text, str) or not text.strip():
            return 0, "Empty text"
            
        # Check content quality
        is_valid, message = analyze_content_quality(text)
        if is_valid:
            cleaned_text = preprocessor.clean_text(text)
            prediction = model.predict([cleaned_text])
            return int(prediction[0]), message
        else:
            # Only mark as fake if clearly not news
            if "too short" in message.lower() or "error" in message.lower():
                return 0, message
            # Otherwise use the model
            cleaned_text = preprocessor.clean_text(text)
            prediction = model.predict([cleaned_text])
            return int(prediction[0]), message

    except Exception as e:
        return 0, f"Prediction error: {str(e)}"

class PredictionRequest(BaseModel):
    text: str  

@app.post("/predict_logistic")
def predict_logistic(user_input: PredictionRequest):
    prediction, message = predict_with_validation(logistic_model, user_input.text, TextPreprocessor())
    return {
        "text": user_input.text,
        "predicted_label": prediction,
        "message": message
    }

@app.post("/predict_naive_bayes")
def predict_naive_bayes(user_input: PredictionRequest):
    prediction, message = predict_with_validation(naive_bayes_model, user_input.text, TextPreprocessor())
    return {
        "text": user_input.text,
        "predicted_label": prediction,
        "message": message
    }

@app.post("/predict_svm")
def predict_svm(user_input: PredictionRequest):
    prediction, message = predict_with_validation(svm_model, user_input.text, TextPreprocessor())
    return {
        "text": user_input.text,
        "predicted_label": prediction,
        "message": message
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8002)