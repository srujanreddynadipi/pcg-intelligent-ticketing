# PCG Intelligent Ticketing System

An AI-powered ITSM (IT Service Management) ticketing system with intelligent ticket categorization and automated resolver routing using machine learning.

## 🎯 Overview

This system uses machine learning models trained on 100,000 tickets to automatically categorize IT support tickets and route them to the appropriate resolver groups with 100% accuracy. It features a modern web interface, RESTful API backend, and production-ready ML models.

## 📁 Project Structure

```
├── backend/              # Node.js/Express REST API
├── frontend/             # Next.js web application
└── ml-model-training/    # Python ML models & training scripts
```

## 🚀 Features

- **AI-Powered Ticket Classification** - Automatically categorizes tickets into 11 categories
- **Intelligent Resolver Routing** - Routes tickets to 7 specialized resolver teams
- **User Authentication** - JWT + Google OAuth support
- **Admin Dashboard** - User management and analytics
- **Real-time Predictions** - Instant ticket categorization with confidence scores
- **Responsive Design** - Dark theme UI built with Next.js and Tailwind CSS

## 🛠 Tech Stack

### Frontend
- **Framework**: Next.js 16 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: Radix UI + shadcn/ui
- **State Management**: React Context API
- **Forms**: React Hook Form + Zod validation

### Backend
- **Runtime**: Node.js
- **Framework**: Express.js
- **Database**: MongoDB with Mongoose ODM
- **Authentication**: JWT, Google OAuth 2.0
- **Security**: bcryptjs password hashing

### ML Models
- **Language**: Python 3.x
- **Algorithm**: Random Forest Classifier (200 trees)
- **Features**: TF-IDF vectorization (2,208 features)
- **Libraries**: scikit-learn, pandas, numpy
- **Deployment**: HuggingFace Hub
- **Accuracy**: 100% on test dataset (20,000 tickets)

## 📋 Prerequisites

- Node.js 18+ 
- Python 3.8+
- MongoDB (local or Atlas)
- pnpm/npm
- Git

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/srujanreddynadipi/pcg-intelligent-ticketing.git
cd pcg-intelligent-ticketing
```

### 2. Backend Setup

```bash
cd backend

# Install dependencies
npm install

# Configure environment
cp .env.example .env
# Edit .env with your MongoDB URI, JWT secret, etc.

# Create admin user (optional)
npm run create-admin

# Start development server
npm run dev
# Server runs on http://localhost:5000
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
pnpm install

# Configure environment
cp .env.example .env.local
# Edit .env.local with API URLs

# Start development server
pnpm dev
# App runs on http://localhost:3000
```

### 4. ML Model Setup (Optional)

```bash
cd ml-model-training

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install scikit-learn pandas numpy huggingface-hub joblib

# Configure environment
cp .env.example .env
# Add your HuggingFace token if uploading models

# Train models (optional - pre-trained models available)
python train_models_improved.py

# Test models
python hackathon_demo_test.py
```

## 🔑 Environment Variables

### Backend (.env)
```env
MONGO_URI=mongodb://localhost:27017/itsm-ticketing
JWT_SECRET=your_jwt_secret
PORT=5000
GOOGLE_CLIENT_ID=your_google_client_id
```

### Frontend (.env.local)
```env
NEXT_PUBLIC_API_URL=http://localhost:5000/api
NEXT_PUBLIC_ML_API_URL=http://localhost:8000
```

### ML Training (.env)
```env
HF_TOKEN=your_huggingface_token
```

## 🎫 Ticket Categories & Resolver Mapping

| Category | Resolver Team | Description |
|----------|--------------|-------------|
| Network | Network Team | Network connectivity, VPN, DNS issues |
| Hardware | Service Desk | PC, printer, peripheral issues |
| Software | App Support | Application installations, updates |
| Access | Service Desk | Account access, permissions |
| Database | DBA Team | Database connectivity, queries |
| Security | Security Ops | Security incidents, vulnerabilities |
| Cloud | Cloud Ops | AWS, Azure, cloud infrastructure |
| DevOps | DevOps Team | CI/CD, deployment issues |
| Email | Service Desk | Email configuration, delivery |
| Monitoring | Cloud Ops | System monitoring, alerts |
| Service Request | Service Desk | General IT service requests |

## 🚦 Getting Started

1. **Start MongoDB** (if running locally)
2. **Start Backend**: `cd backend && npm run dev`
3. **Start Frontend**: `cd frontend && pnpm dev`
4. **Access Application**: http://localhost:3000
5. **Login/Signup** to create tickets
6. **Create Ticket** - Watch AI automatically categorize and route it!

## 📊 ML Model Performance

- **Training Accuracy**: 100%
- **Test Accuracy**: 100%
- **Training Dataset**: 100,000 tickets (perfectly balanced)
- **Test Dataset**: 20,000 tickets
- **Real-world Test**: 22/22 correct predictions (100%)
- **Feature Dimensions**: 2,208 (TF-IDF + categorical + keywords)

## 📝 API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/auth/google` - Google OAuth login

### Tickets
- `GET /api/tickets` - Get all tickets
- `POST /api/tickets` - Create new ticket
- `GET /api/tickets/:id` - Get ticket by ID
- `PUT /api/tickets/:id` - Update ticket
- `DELETE /api/tickets/:id` - Delete ticket

### Admin
- `GET /api/admin/users` - Get all users (admin only)
- `PUT /api/admin/users/:id/role` - Update user role (admin only)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- Srujan Reddy Nadipi

## 🙏 Acknowledgments

- HuggingFace for model hosting
- scikit-learn for ML algorithms
- Next.js and Vercel for frontend framework
- MongoDB for database
