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
    """Enhanced check for structured news articles"""
    try:
        score = 0
        confidence = 0
        reasons = []

        number_patterns = [
            r'\$\d+(?:\.\d+)?(?:\s*(?:billion|million|thousand))?',  
            r'\d+(?:\.\d+)?%',  
            r'\d+(?:\.\d+)?(?:\s*(?:billion|million|thousand))', 
            r'\d{2,3}(?:,\d{3})*\.\d{2}'
        ]
        stats_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in number_patterns)
        if stats_count >= 2:
            score += 3
            confidence += 0.2
            reasons.append("Contains statistical data")

        if len(text.split('\n\n')) >= 3: 
            score += 2
            confidence += 0.1
            reasons.append("Proper news formatting")

        quote_count = len(re.findall(r'"[^"]*"', text))
        if quote_count >= 2:
            score += 3
            confidence += 0.15
            reasons.append("Contains multiple quotes")

        official_patterns = [
            r'according to',
            r'said',
            r'stated',
            r'announced',
            r'reported',
            r'official',
            r'minister',
            r'department',
            r'government'
        ]
        source_count = sum(len(re.findall(pattern, text.lower())) for pattern in official_patterns)
        if source_count >= 2:
            score += 2
            confidence += 0.15
            reasons.append("Contains official sources")

        location_org_patterns = [
            r'in [A-Z][a-z]+',  
            r'at [A-Z][a-z]+',
            r'[A-Z][a-z]+ (?:Corporation|Inc\.|Ltd\.|Department|Ministry)'  
        ]
        entity_count = sum(len(re.findall(pattern, text)) for pattern in location_org_patterns)
        if entity_count >= 2:
            score += 2
            confidence += 0.1
            reasons.append("Contains locations and organizations")

        date_patterns = [
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
            r'on (?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)',
            r'last (?:week|month|year)',
            r'in (?:FY|Q)\d{2,4}'
        ]
        if any(re.search(pattern, text) for pattern in date_patterns):
            score += 2
            confidence += 0.1
            reasons.append("Contains temporal references")

        final_confidence = min(confidence, 1.0)
        
        return {
            'is_news': score >= 7, 
            'score': score,
            'confidence': final_confidence,
            'reasons': reasons
        }
    except Exception as e:
        logger.error(f"Error in news pattern check: {e}")
        return {'is_news': False, 'score': 0, 'confidence': 0, 'reasons': ['Error in analysis']}

def predict_with_validation(model, text, preprocessor):
    """Enhanced prediction with adjusted confidence threshold"""
    try:
        if not isinstance(text, str) or not text.strip():
            return 0, "Empty text", 0.0
            
        news_check = check_news_patterns(text)
        confidence = news_check['confidence'] * 100  
        
        prediction = 1 if confidence >= 40 else 0
        reasons = "\n".join(news_check['reasons']) if news_check['reasons'] else "Analysis incomplete"
            
        return prediction, reasons, confidence

    except Exception as e:
        return 0, f"Prediction error: {str(e)}", 0.0

class PredictionRequest(BaseModel):
    text: str  

@app.post("/predict_logistic")
def predict_logistic(user_input: PredictionRequest):
    prediction, message, confidence = predict_with_validation(logistic_model, user_input.text, TextPreprocessor())
    return {
        "text": user_input.text,
        "predicted_label": prediction,
        "message": message,
        "confidence": confidence
    }

@app.post("/predict_naive_bayes")
def predict_naive_bayes(user_input: PredictionRequest):
    prediction, message, confidence = predict_with_validation(naive_bayes_model, user_input.text, TextPreprocessor())
    return {
        "text": user_input.text,
        "predicted_label": prediction,
        "message": message,
        "confidence": confidence
    }

@app.post("/predict_svm")
def predict_svm(user_input: PredictionRequest):
    prediction, message, confidence = predict_with_validation(svm_model, user_input.text, TextPreprocessor())
    return {
        "text": user_input.text,
        "predicted_label": prediction,
        "message": message,
        "confidence": confidence
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8002)