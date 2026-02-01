"""
Text preprocessing utilities for the chatbot.
"""

import re
import string
import json
import pickle
import numpy as np
from typing import List, Tuple, Dict
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)


class TextPreprocessor:
    """Handles all text preprocessing for the chatbot."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.ignore_chars = set(string.punctuation)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return word_tokenize(text)
    
    def lemmatize(self, words: List[str]) -> List[str]:
        """Lemmatize a list of words."""
        return [self.lemmatizer.lemmatize(word.lower()) 
                for word in words 
                if word not in self.ignore_chars]
    
    def process(self, text: str) -> List[str]:
        """Full preprocessing pipeline."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        lemmatized = self.lemmatize(tokens)
        return lemmatized


class Vocabulary:
    """Manages vocabulary for the chatbot."""
    
    def __init__(self):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.words: List[str] = []
        self.classes: List[str] = []
        self.class2idx: Dict[str, int] = {}
        self.idx2class: Dict[int, str] = {}
    
    def build_vocab(self, words: List[str], classes: List[str]):
        """Build vocabulary from words and classes."""
        # Sort and store unique words
        self.words = sorted(set(words))
        self.word2idx = {word: idx for idx, word in enumerate(self.words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        # Sort and store classes
        self.classes = sorted(set(classes))
        self.class2idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx2class = {idx: cls for cls, idx in self.class2idx.items()}
    
    def text_to_bow(self, words: List[str]) -> np.ndarray:
        """Convert words to bag-of-words vector."""
        bow = np.zeros(len(self.words), dtype=np.float32)
        for word in words:
            if word in self.word2idx:
                bow[self.word2idx[word]] = 1
        return bow
    
    def class_to_idx(self, class_name: str) -> int:
        """Convert class name to index."""
        return self.class2idx.get(class_name, -1)
    
    def idx_to_class(self, idx: int) -> str:
        """Convert index to class name."""
        return self.idx2class.get(idx, "unknown")
    
    def save(self, filepath: str):
        """Save vocabulary to file."""
        data = {
            'words': self.words,
            'classes': self.classes,
            'word2idx': self.word2idx,
            'class2idx': self.class2idx
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load vocabulary from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.words = data['words']
        self.classes = data['classes']
        self.word2idx = data['word2idx']
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.class2idx = data['class2idx']
        self.idx2class = {idx: cls for cls, idx in self.class2idx.items()}
    
    @property
    def vocab_size(self) -> int:
        return len(self.words)
    
    @property
    def num_classes(self) -> int:
        return len(self.classes)


def load_intents(filepath: str) -> Dict:
    """Load intents from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def prepare_training_data(intents_path: str) -> Tuple[np.ndarray, np.ndarray, Vocabulary, Dict]:
    """
    Prepare training data from intents file.
    
    Returns:
        X: Training features (bag of words)
        y: Training labels (one-hot encoded)
        vocab: Vocabulary object
        intents: Original intents dictionary
    """
    preprocessor = TextPreprocessor()
    vocab = Vocabulary()
    
    # Load intents
    intents = load_intents(intents_path)
    
    all_words = []
    classes = []
    documents = []  # (words, tag) pairs
    
    # Process each intent
    for intent in intents['intents']:
        tag = intent['tag']
        classes.append(tag)
        
        for pattern in intent['patterns']:
            words = preprocessor.process(pattern)
            all_words.extend(words)
            documents.append((words, tag))
    
    # Build vocabulary
    vocab.build_vocab(all_words, classes)
    
    # Create training data
    X = []
    y = []
    
    for words, tag in documents:
        # Bag of words
        bow = vocab.text_to_bow(words)
        X.append(bow)
        
        # One-hot encode the label
        label = np.zeros(vocab.num_classes, dtype=np.float32)
        label[vocab.class_to_idx(tag)] = 1
        y.append(label)
    
    return np.array(X), np.array(y), vocab, intents
