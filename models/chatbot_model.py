"""
Neural Network model for chatbot intent classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChatBotNN(nn.Module):
    """
    Neural Network for intent classification.
    Architecture: Input -> FC -> ReLU -> Dropout -> FC -> ReLU -> Dropout -> FC -> Softmax
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout: float = 0.5):
        super(ChatBotNN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x


class ChatBotLSTM(nn.Module):
    """
    LSTM-based model for intent classification.
    Better for capturing sequential patterns in text.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, 
                 num_classes: int, num_layers: int = 2, dropout: float = 0.5):
        super(ChatBotLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        
        output = self.fc(hidden)
        return output


class IntentClassifier:
    """Wrapper class for model inference."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def predict(self, x: torch.Tensor) -> tuple:
        """
        Predict intent for input.
        
        Returns:
            predicted_class: Index of predicted class
            confidence: Confidence score
            probabilities: All class probabilities
        """
        with torch.no_grad():
            x = x.to(self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            outputs = self.model(x)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            return predicted.item(), confidence.item(), probabilities.squeeze().cpu().numpy()
    
    def predict_top_k(self, x: torch.Tensor, k: int = 3) -> list:
        """
        Get top-k predictions.
        
        Returns:
            List of (class_index, confidence) tuples
        """
        with torch.no_grad():
            x = x.to(self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            outputs = self.model(x)
            probabilities = F.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, k, dim=1)
            
            results = []
            for prob, idx in zip(top_probs.squeeze().cpu().numpy(), 
                                  top_indices.squeeze().cpu().numpy()):
                results.append((int(idx), float(prob)))
            
            return results
