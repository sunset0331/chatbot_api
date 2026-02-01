"""
Training script for the chatbot model.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.chatbot_model import ChatBotNN
from utils.preprocessing import prepare_training_data


def train_model(
    intents_path: str,
    model_save_path: str,
    vocab_save_path: str,
    hidden_size: int = 128,
    learning_rate: float = 0.001,
    epochs: int = 200,
    batch_size: int = 8,
    dropout: float = 0.5,
    device: str = None
):
    """Train the chatbot model."""
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Prepare data
    print("Loading and preprocessing data...")
    X, y, vocab, intents = prepare_training_data(intents_path)
    
    print(f"Vocabulary size: {vocab.vocab_size}")
    print(f"Number of classes: {vocab.num_classes}")
    print(f"Training samples: {len(X)}")
    print(f"Classes: {vocab.classes}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
    )
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = ChatBotNN(
        input_size=vocab.vocab_size,
        hidden_size=hidden_size,
        num_classes=vocab.num_classes,
        dropout=dropout
    ).to(device)
    
    print(f"\nModel architecture:\n{model}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0
    early_stop_patience = 30
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.argmax(dim=1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y.argmax(dim=1)).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.argmax(dim=1))
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y.argmax(dim=1)).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_acc'].append(val_acc)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'vocab_size': vocab.vocab_size,
                'num_classes': vocab.num_classes,
                'hidden_size': hidden_size,
                'dropout': dropout
            }, model_save_path)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Save vocabulary
    vocab.save(vocab_save_path)
    
    # Save intents for response generation
    intents_save_path = os.path.join(os.path.dirname(vocab_save_path), 'intents_processed.json')
    with open(intents_save_path, 'w') as f:
        json.dump(intents, f)
    
    # Save training history
    history_path = os.path.join(os.path.dirname(model_save_path), 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f)
    
    print(f"\n{'='*50}")
    print("Training Complete!")
    print(f"{'='*50}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {model_save_path}")
    print(f"Vocabulary saved to: {vocab_save_path}")
    
    return model, vocab, training_history


def main():
    parser = argparse.ArgumentParser(description='Train the chatbot model')
    parser.add_argument('--intents', type=str, default='data/intents.json',
                        help='Path to intents JSON file')
    parser.add_argument('--model-path', type=str, default='models/chatbot_model.pth',
                        help='Path to save the trained model')
    parser.add_argument('--vocab-path', type=str, default='models/vocab.pkl',
                        help='Path to save the vocabulary')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Hidden layer size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    args = parser.parse_args()
    
    # Get absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    intents_path = os.path.join(base_dir, args.intents)
    model_path = os.path.join(base_dir, args.model_path)
    vocab_path = os.path.join(base_dir, args.vocab_path)
    
    # Create directories if needed
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Train
    train_model(
        intents_path=intents_path,
        model_save_path=model_path,
        vocab_save_path=vocab_path,
        hidden_size=args.hidden_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout=args.dropout
    )


if __name__ == '__main__':
    main()
