import joblib
import nltk
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_nltk_resources():
    """Verify NLTK resources are available"""
    required_resources = [
        'stopwords', 'punkt', 'wordnet', 
        'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'
    ]
    
    for resource in required_resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' 
                          else f'corpora/{resource}')
            logger.info(f"✓ {resource} is available")
        except LookupError:
            logger.warning(f"× {resource} not found, downloading...")
            nltk.download(resource)

def verify_models():
    """Verify ML models are available"""
    model_paths = {
        'logistic': r"C:\Users\Lenovo\OneDrive\Desktop\Truthtell_Tech4Stack\logistic_model.pkl",
        'naive_bayes': r"C:\Users\Lenovo\OneDrive\Desktop\Truthtell_Tech4Stack\naive_bayes_model.pkl",
        'svm': r"C:\Users\Lenovo\OneDrive\Desktop\Truthtell_Tech4Stack\svm_model.pkl"
    }
    
    for name, path in model_paths.items():
        if Path(path).exists():
            try:
                model = joblib.load(path)
                logger.info(f"✓ {name} model loaded successfully")
            except Exception as e:
                logger.error(f"× Error loading {name} model: {e}")
        else:
            logger.error(f"× {name} model not found at {path}")

if __name__ == "__main__":
    logger.info("Starting system verification...")
    verify_nltk_resources()
    verify_models()
    logger.info("Verification complete")
