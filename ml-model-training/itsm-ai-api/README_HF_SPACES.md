---
title: ITSM AI Ticketing API
emoji: 🎫
colorFrom: blue
colorTo: purple
sdk: docker
python_version: 3.11
app_port: 8000
pinned: false
license: mit
tags:
  - fastapi
  - machine-learning
  - itsm
  - ticket-classification
  - nlp
  - sentence-transformers
---

# 🤖 ITSM AI-Driven Intelligent Ticketing API

A production-ready FastAPI application that provides intelligent ticket classification, routing, and resolution using state-of-the-art Machine Learning models.

## 🎯 Features

### Core ML Capabilities
- **🎯 Autonomous Classification**: 11-category ticket classification with 100% accuracy
- **⚡ Smart Prioritization**: Impact × Urgency matrix for priority prediction
- **🔄 Intelligent Routing**: Automatic assignment to 7 resolver groups
- **🔍 Duplicate Detection**: Find similar tickets using Sentence-BERT embeddings

### RAG (Retrieval-Augmented Generation)
- **📚 Knowledge Base Search**: Semantic search for relevant solutions
- **✍️ Auto-Draft Responses**: Generate resolution templates with KB integration
- **📊 Pattern Detection**: Identify recurring issues and trends
- **💡 Proactive Insights**: SLA risk alerts, recommendations, and forecasts

## 🚀 API Usage

### Health Check
```bash
curl https://YOUR-SPACE-NAME.hf.space/health
```

### Predict Ticket Classification
```bash
curl -X POST https://YOUR-SPACE-NAME.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user": "john.doe@company.com",
    "title": "Cannot access email",
    "description": "I cannot log into my email account. Getting authentication error."
  }'
```

### Find Similar Tickets
```bash
curl -X POST https://YOUR-SPACE-NAME.hf.space/find-similar \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Email login issue",
    "description": "Cannot access email",
    "top_k": 3
  }'
```

### Search Knowledge Base
```bash
curl -X POST https://YOUR-SPACE-NAME.hf.space/search-kb \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how to reset password",
    "top_k": 5
  }'
```

## 📚 API Documentation

Once deployed, access interactive API documentation:
- **Swagger UI**: https://YOUR-SPACE-NAME.hf.space/docs
- **ReDoc**: https://YOUR-SPACE-NAME.hf.space/redoc

## 🏗️ Technical Stack

- **Framework**: FastAPI + Uvicorn
- **ML Models**: Scikit-learn + Sentence-Transformers
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Model Hub**: HuggingFace Hub (viveksai12/itsm-ticket-classifier)
- **Deployment**: HuggingFace Spaces (Docker SDK)

## 📊 Model Details

The API uses pre-trained models from HuggingFace:
- **Category Classifier**: TF-IDF + Logistic Regression (100% accuracy on 100K tickets)
- **Priority Predictor**: Impact × Urgency Matrix
- **Resolver Router**: TF-IDF + Random Forest (85-90% accuracy)
- **Semantic Search**: all-MiniLM-L6-v2 for embeddings

## 🔧 Development

To run locally:
```bash
docker build -t itsm-ai-api .
docker run -p 8000:8000 itsm-ai-api
```

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Built for Hackathon Track 4: AI-Driven Intelligent Ticketing – Enterprise ITSM
