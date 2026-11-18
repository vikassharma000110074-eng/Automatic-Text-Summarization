
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Initialize stemmer
stemmer = PorterStemmer()

# Load English stopwords
STOP_WORDS = set(ENGLISH_STOP_WORDS)


def preprocess_text(text, remove_stopwords=True, apply_stemming=False):
    """
    Preprocess text by:
    1. Converting text to lowercase
    2. Removing special characters
    3. Tokenizing text
    4. Removing stopwords
    5. Applying stemming (optional)
    """

    if not isinstance(text, str):
        return []

    # Convert to lowercase
    text = text.lower()

    # Extract words
    words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)

    # Remove stopwords
    if remove_stopwords:
        words = [
            word for word in words
            if word not in STOP_WORDS
        ]

    # Apply stemming if requested
    if apply_stemming:
        words = [
            stemmer.stem(word)
            for word in words
        ]

    return words
