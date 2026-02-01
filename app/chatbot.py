"""
Chatbot Engine - Handles inference and response generation.
"""

import os
import json
import random
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.chatbot_model import ChatBotNN, IntentClassifier
from utils.preprocessing import TextPreprocessor, Vocabulary


class ChatBot:
    """
    Main chatbot class that handles:
    - Loading trained model
    - Processing user input
    - Generating responses
    """
    
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        intents_path: str,
        confidence_threshold: float = 0.25,
        device: str = None
    ):
        self.confidence_threshold = confidence_threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor()
        
        # Load vocabulary
        self.vocab = Vocabulary()
        self.vocab.load(vocab_path)
        
        # Load intents
        with open(intents_path, 'r') as f:
            self.intents = json.load(f)
        
        # Create intent lookup
        self.intent_responses = {}
        for intent in self.intents['intents']:
            self.intent_responses[intent['tag']] = intent['responses']
        
        # Load model
        self.model = self._load_model(model_path)
        self.classifier = IntentClassifier(self.model, self.device)
        
        # Conversation history
        self.conversation_history: List[Dict] = []
        
        print(f"ChatBot initialized on {self.device}")
        print(f"Vocabulary size: {self.vocab.vocab_size}")
        print(f"Number of intents: {self.vocab.num_classes}")
    
    def _load_model(self, model_path: str) -> ChatBotNN:
        """Load the trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = ChatBotNN(
            input_size=checkpoint['vocab_size'],
            hidden_size=checkpoint['hidden_size'],
            num_classes=checkpoint['num_classes'],
            dropout=checkpoint['dropout']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def preprocess(self, text: str) -> torch.Tensor:
        """Preprocess text for model input."""
        words = self.preprocessor.process(text)
        bow = self.vocab.text_to_bow(words)
        return torch.FloatTensor(bow)
    
    def predict_intent(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict the intent of user input.
        
        Returns:
            intent: Predicted intent tag
            confidence: Confidence score
            all_intents: Dictionary of all intents with their scores
        """
        # Preprocess
        input_tensor = self.preprocess(text)
        
        # Predict
        predicted_idx, confidence, probabilities = self.classifier.predict(input_tensor)
        
        # Get intent name
        intent = self.vocab.idx_to_class(predicted_idx)
        
        # Get all intent scores
        all_intents = {}
        for idx, prob in enumerate(probabilities):
            intent_name = self.vocab.idx_to_class(idx)
            all_intents[intent_name] = float(prob)
        
        return intent, confidence, all_intents
    
    def get_response(self, intent: str) -> str:
        """Get a random response for the given intent."""
        if intent in self.intent_responses:
            return random.choice(self.intent_responses[intent])
        return "I'm not sure how to respond to that. Could you rephrase?"
    
    def chat(self, user_message: str) -> Dict:
        """
        Process user message and generate response.
        
        Returns:
            Dictionary containing:
            - response: Bot response text
            - intent: Detected intent
            - confidence: Confidence score
            - understood: Whether the bot understood the message
        """
        # Predict intent
        intent, confidence, all_intents = self.predict_intent(user_message)
        
        # Check confidence threshold
        understood = confidence >= self.confidence_threshold
        
        if understood:
            response = self.get_response(intent)
        else:
            response = "I'm sorry, I didn't quite understand that. Could you please rephrase?"
            intent = "unknown"
        
        # Create result
        result = {
            "response": response,
            "intent": intent,
            "confidence": round(confidence, 4),
            "understood": understood,
            "top_intents": dict(sorted(all_intents.items(), key=lambda x: x[1], reverse=True)[:3])
        }
        
        # Store in history
        self.conversation_history.append({
            "user": user_message,
            "bot": response,
            "intent": intent,
            "confidence": confidence
        })
        
        return result
    
    def get_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_available_intents(self) -> List[str]:
        """Get list of available intents."""
        return self.vocab.classes


# Singleton instance for API
_chatbot_instance: Optional[ChatBot] = None


def get_chatbot() -> ChatBot:
    """Get or create chatbot instance."""
    global _chatbot_instance
    
    if _chatbot_instance is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        model_path = os.path.join(base_dir, 'models', 'chatbot_model.pth')
        vocab_path = os.path.join(base_dir, 'models', 'vocab.pkl')
        intents_path = os.path.join(base_dir, 'models', 'intents_processed.json')
        
        _chatbot_instance = ChatBot(
            model_path=model_path,
            vocab_path=vocab_path,
            intents_path=intents_path
        )
    
    return _chatbot_instance


def reset_chatbot():
    """Reset the chatbot instance."""
    global _chatbot_instance
    _chatbot_instance = None
