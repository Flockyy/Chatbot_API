#  Chatbot API
<div align="center">

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlalchemy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NLTK](https://img.shields.io/badge/NLTK-154f3c?style=for-the-badge&logo=python&logoColor=white)](https://www.nltk.org/)

**Production-ready RESTful API for intelligent conversational chatbot with NLP capabilities**

[Features](#-key-features)  [Architecture](#-architecture)  [Installation](#-installation)  [API Docs](#-api-documentation)  [Usage](#-usage)  [Development](#-development)

</div>

---

##  Overview

A full-stack **FastAPI-based chatbot system** with custom NLP models, database persistence, and comprehensive REST endpoints. Implements question-answering capabilities using PyTorch neural networks, NLTK text processing, and SQLAlchemy ORM for conversation management.

###  Key Features

- **Custom NLP Model**: Seq2seq architecture with attention mechanism for response generation
- **FastAPI Backend**: High-performance async API with automatic OpenAPI documentation
- **Database Persistence**: SQLAlchemy ORM with Alembic migrations for conversation history
- **Intent Classification**: NLTK-powered tokenization and stemming for query understanding
- **CamemBERT Integration**: Pre-trained French language model support (notebooks)
- **Test Coverage**: Comprehensive test suite with pytest
- **Production-Ready**: Includes health checks, pre-start scripts, and linting
- **API Versioning**: Organized v1 API structure for maintainability

---

##  Architecture

```

           FastAPI Application               
                                             
     
        API v1 Endpoints                   
    /chats  /questions  /answers           
     
                                            
     
           CRUD Operations                 
    (Business Logic Layer)                 
     
                                            
     
        SQLAlchemy ORM Models              
    Chat | Question | Answer               
     
                                            
     
           SQLite Database                 
    (sql.db)                               
     

                     
                     
       
           AI Module (ai/)        
          NLP Utils (NLTK)       
          PyTorch Model          
          Seq2seq Training       
       
```

---

##  Technologies

| Category | Tools |
|----------|-------|
| **Framework** | FastAPI, Uvicorn |
| **Database** | SQLite, SQLAlchemy 2.0, Alembic |
| **NLP** | PyTorch, NLTK, CamemBERT (HuggingFace) |
| **Validation** | Pydantic v2 |
| **Testing** | Pytest, Coverage |
| **Code Quality** | Flake8, MyPy, Black |
| **Python** | 3.9+ |

---

##  Project Structure

```
Chatbot_API/
 src/
    main.py                  # FastAPI application entry point
    api/
       api_v1/              # API version 1 endpoints
          endpoints/
             chats.py     # Chat CRUD endpoints
             questions.py # Question management
             answers.py   # Answer generation
          api.py           # API router aggregation
       deps.py              # Dependency injection
    core/
       config.py            # Application configuration
    crud/
       base.py              # Generic CRUD operations
       crud_chat.py         # Chat-specific operations
       crud_question.py     # Question operations
       crud_answer.py       # Answer operations
    db/
       base.py              # Database models registry
       base_class.py        # Declarative base
       session.py           # Database session factory
       init_db.py           # DB initialization
    models/
       chat.py              # Chat SQLAlchemy model
       question.py          # Question model
       answer.py            # Answer model
    schemas/
       chat.py              # Chat Pydantic schemas
       question.py          # Question schemas
       answer.py            # Answer schemas
    tests/                   # Test suite
        api/                 # API endpoint tests
        crud/                # CRUD operation tests
        utils/               # Test utilities
 ai/
    model.py                 # PyTorch neural network architecture
    train.py                 # Model training script
    seq2seq.py               # Sequence-to-sequence implementation
    nltk_utils.py            # Tokenization, stemming, bag-of-words
    create_model_db.py       # Generate training data
 notebooks/
    camembert-pretrained-qna.ipynb  # CamemBERT fine-tuning
    data_processing.ipynb           # Data exploration
 alembic/                     # Database migrations
 scripts/
    test.sh                  # Run tests
    test-cov-html.sh         # Coverage report
    lint.sh                  # Code quality checks
 alembic.ini                  # Alembic configuration
 backend_pre_start.py         # Pre-start health checks
 initial_data.py              # Seed database
 sql.db                       # SQLite database file
```

---

##  Installation

### Prerequisites

- Python 3.9+
- pip or conda
- Git

### Setup

1. **Clone the repository**
```ash
git clone https://github.com/Flockyy/Chatbot_API.git
cd Chatbot_API
```

2. **Create virtual environment**
```ash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**
```ash
pip install -r requirements.txt
```

4. **Download NLTK data**
```ash
python -c "import nltk; nltk.download('punkt')"
```

5. **Initialize database**
```ash
alembic upgrade head
python initial_data.py
```

6. **Train NLP model (optional)**
```ash
cd ai
python train.py
```

---

##  Usage

### Starting the API Server

```ash
# Development mode with auto-reload
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Server will be available at **http://localhost:8000**

### API Documentation

Once running, access interactive API docs:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

##  API Documentation

### Core Endpoints

#### Chat Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| \POST\ | \/api/v1/chats/\ | Create new chat session |
| \GET\ | \/api/v1/chats/{id}\ | Get chat by ID |
| \GET\ | \/api/v1/chats/\ | List all chats |
| \DELETE\ | \/api/v1/chats/{id}\ | Delete chat |

#### Question Handling

| Method | Endpoint | Description |
|--------|----------|-------------|
| \POST\ | \/api/v1/questions/\ | Submit new question |
| \GET\ | \/api/v1/questions/{id}\ | Get question details |
| \GET\ | \/api/v1/questions/chat/{chat_id}\ | Get all questions in chat |

#### Answer Generation

| Method | Endpoint | Description |
|--------|----------|-------------|
| \GET\ | \/api/v1/answers/{question_id}\ | Generate answer for question |
| \GET\ | \/api/v1/answers/\ | List all answers |

### Example Requests

<details>
<summary><b>Create Chat & Ask Question</b> (Click to expand)</summary>

```ash
# 1. Create chat session
curl -X POST http://localhost:8000/api/v1/chats/ \\
  -H "Content-Type: application/json" \\
  -d '{"title": "Technical Support"}'

# Response: {"id": 1, "title": "Technical Support", "created_at": "2024-01-15T10:00:00"}

# 2. Ask question
curl -X POST http://localhost:8000/api/v1/questions/ \\
  -H "Content-Type: application/json" \\
  -d '{"chat_id": 1, "text": "How do I reset my password?"}'

# 3. Get answer
curl -X GET http://localhost:8000/api/v1/answers/1
```

</details>

<details>
<summary><b>Python Client Example</b> (Click to expand)</summary>

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# Create chat
chat = requests.post(f"{BASE_URL}/chats/", json={"title": "Support"})
chat_id = chat.json()["id"]

# Ask question
question = requests.post(
    f"{BASE_URL}/questions/", 
    json={"chat_id": chat_id, "text": "What are your business hours?"}
)
question_id = question.json()["id"]

# Get answer
answer = requests.get(f"{BASE_URL}/answers/{question_id}")
print(answer.json()["text"])
```

</details>

---

##  NLP Model Details

### Architecture

**Seq2seq with Attention Mechanism**:
- Embedding layer (300-dim word vectors)
- Bidirectional LSTM encoder
- Attention-based decoder
- Softmax output layer

### Training

```ash
cd ai
python train.py --epochs 1000 --batch-size 8 --learning-rate 0.001
```

Training data format in \create_model_db.py\:
```python
intents = {
    "greetings": {
        "patterns": ["hello", "hi", "hey"],
        "responses": ["Hello!", "Hi there!", "Greetings!"]
    },
    # ... more intents
}
```

---

##  Development

### Running Tests

```ash
# All tests
pytest src/tests/

# With coverage
./scripts/test-cov-html.sh
# Open htmlcov/index.html

# Specific test file
pytest src/tests/api/test_chats.py -v
```

### Code Quality

```ash
# Linting
flake8 src/
./scripts/lint.sh

# Type checking
mypy src/

# Auto-formatting (if Black installed)
black src/
```

### Database Migrations

```ash
# Create migration
alembic revision --autogenerate -m "Add new column"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

---

##  Docker Deployment (Future)

```dockerfile
# Example Dockerfile structure
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

##  Contributing

1. Fork the repository
2. Create feature branch (\git checkout -b feature/new-feature\)
3. Run tests: \pytest src/tests/\
4. Run linter: \lake8 src/\
5. Commit changes (\git commit -m 'Add feature'\)
6. Push to branch (\git push origin feature/new-feature\)
7. Open Pull Request

---

##  License

This project is open-source and available under the MIT License.

---

##  Acknowledgments

- FastAPI framework by Sebastián Ramírez
- PyTorch team for deep learning tools
- NLTK community for NLP utilities
- CamemBERT model by Inria

---

<div align="center">

**[ Back to Top](#-chatbot-api)**

Made with  by [Florian Abgrall](https://github.com/Flockyy)

</div>
