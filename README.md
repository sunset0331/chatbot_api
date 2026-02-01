# 🤖 AI Chatbot API

A production-ready AI Chatbot built with **PyTorch** and **FastAPI**. Uses deep learning for intent classification and natural language understanding.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **Deep Learning Model**: Neural network for intent classification
- **Natural Language Processing**: Text preprocessing with NLTK
- **REST API**: FastAPI with automatic OpenAPI documentation
- **Web Interface**: Beautiful chat UI included
- **Docker Ready**: Full containerization support
- **Production Ready**: Health checks, CORS, error handling

## 📁 Project Structure

```
chatbot_api/
├── app/
│   ├── main.py           # FastAPI application
│   └── chatbot.py        # Chatbot engine
├── data/
│   └── intents.json      # Training data
├── models/
│   └── chatbot_model.py  # Neural network architecture
├── training/
│   └── train.py          # Training script
├── utils/
│   └── preprocessing.py  # Text preprocessing
├── static/
│   └── index.html        # Web chat interface
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd chatbot_api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python training/train.py
```

Expected output:
```
Using device: cpu
Vocabulary size: 85
Number of classes: 15
Training samples: 95
...
Training Complete!
Best Validation Accuracy: 95.00%
```

### 3. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open in Browser

- **Chat Interface**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/docs
- **API Docs (ReDoc)**: http://localhost:8000/redoc

## 🐳 Docker Deployment

### Build and Run

```bash
# Train first (if not done)
python training/train.py

# Build and run with Docker Compose
docker-compose up --build

# Or use Docker directly
docker build -t chatbot-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models chatbot-api
```

### Train in Docker

```bash
docker-compose --profile training run chatbot-train
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web chat interface |
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Send message, get response |
| `GET` | `/intents` | List available intents |
| `GET` | `/history` | Get conversation history |
| `DELETE` | `/history` | Clear history |
| `POST` | `/reset` | Reset chatbot |

### Example API Usage

```bash
# Send a message
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'

# Response
{
  "response": "Hello! How can I help you today?",
  "intent": "greeting",
  "confidence": 0.9823,
  "understood": true,
  "top_intents": {
    "greeting": 0.9823,
    "about": 0.0089,
    "help": 0.0045
  },
  "timestamp": "2025-01-22T10:30:00"
}
```

### Python Client Example

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"message": "Tell me a joke"}
)
print(response.json()["response"])
```

## 🎯 Adding Custom Intents

Edit `data/intents.json` to add new intents:

```json
{
  "intents": [
    {
      "tag": "custom_intent",
      "patterns": [
        "Pattern 1",
        "Pattern 2",
        "More variations..."
      ],
      "responses": [
        "Response 1",
        "Response 2"
      ]
    }
  ]
}
```

Then retrain:
```bash
python training/train.py
```

## 🧠 Model Architecture

```
ChatBotNN(
  (fc1): Linear(vocab_size -> 128)
  (batch_norm1): BatchNorm1d(128)
  (fc2): Linear(128 -> 64)
  (batch_norm2): BatchNorm1d(64)
  (fc3): Linear(64 -> num_classes)
  (dropout): Dropout(p=0.5)
)
```

## 📊 Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 200 | Training epochs |
| `--batch-size` | 8 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--hidden-size` | 128 | Hidden layer size |
| `--dropout` | 0.5 | Dropout rate |

```bash
python training/train.py --epochs 300 --lr 0.0005 --hidden-size 256
```

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# Test the API manually
python -c "
from app.chatbot import get_chatbot
bot = get_chatbot()
print(bot.chat('Hello!'))
"
```

## 🚀 Production Deployment

### Using Gunicorn

```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | 8000 |
| `HOST` | Server host | 0.0.0.0 |
| `WORKERS` | Gunicorn workers | 4 |

## 📝 License

MIT License - feel free to use for personal or commercial projects.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a Pull Request

---

Built with ❤️ using PyTorch and FastAPI
