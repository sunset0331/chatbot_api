"""
Tests for the Chatbot API.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPreprocessing:
    """Test preprocessing utilities."""
    
    def test_text_preprocessor(self):
        from utils.preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        
        # Test cleaning
        assert preprocessor.clean_text("Hello, World!") == "hello world"
        assert preprocessor.clean_text("  Multiple   Spaces  ") == "multiple spaces"
        
        # Test tokenization
        tokens = preprocessor.tokenize("hello world")
        assert len(tokens) == 2
        
        # Test full pipeline
        result = preprocessor.process("Hello, how are you?")
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_vocabulary(self):
        from utils.preprocessing import Vocabulary
        import numpy as np
        
        vocab = Vocabulary()
        words = ["hello", "world", "test", "hello"]
        classes = ["greeting", "farewell"]
        
        vocab.build_vocab(words, classes)
        
        assert vocab.vocab_size == 3  # unique words
        assert vocab.num_classes == 2
        assert "hello" in vocab.word2idx
        
        # Test bag of words
        bow = vocab.text_to_bow(["hello", "world"])
        assert isinstance(bow, np.ndarray)
        assert bow.sum() == 2


class TestModel:
    """Test model architecture."""
    
    def test_chatbot_nn(self):
        import torch
        from models.chatbot_model import ChatBotNN
        
        model = ChatBotNN(
            input_size=100,
            hidden_size=64,
            num_classes=10,
            dropout=0.5
        )
        
        # Test forward pass
        x = torch.randn(4, 100)
        output = model(x)
        
        assert output.shape == (4, 10)
    
    def test_intent_classifier(self):
        import torch
        from models.chatbot_model import ChatBotNN, IntentClassifier
        
        model = ChatBotNN(
            input_size=50,
            hidden_size=32,
            num_classes=5
        )
        
        classifier = IntentClassifier(model, device='cpu')
        
        x = torch.randn(50)
        predicted, confidence, probs = classifier.predict(x)
        
        assert isinstance(predicted, int)
        assert 0 <= confidence <= 1
        assert len(probs) == 5


# Integration tests (require trained model)
class TestAPIIntegration:
    """Integration tests for the API (requires trained model)."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        try:
            from app.main import app
            return TestClient(app)
        except Exception:
            pytest.skip("Model not trained yet")
    
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
    
    def test_chat_endpoint(self, client):
        response = client.post(
            "/chat",
            json={"message": "Hello"}
        )
        
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "intent" in data
        assert "confidence" in data
    
    def test_intents_endpoint(self, client):
        response = client.get("/intents")
        
        if response.status_code == 500:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
        data = response.json()
        assert "intents" in data
        assert "count" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
