# Backend PCG - ITSM Ticket Management System

A Node.js/Express backend API for managing ITSM tickets with user authentication, ticket routing, and admin dashboard functionality.

## Features

- 🔐 User Authentication (JWT + Google OAuth)
- 🎫 Ticket Management (Create, Read, Update, Delete)
- 🔄 Intelligent Ticket Routing
- 👥 User Management
- 🛡️ Admin Dashboard
- 📊 Analytics and Reporting

## Tech Stack

- **Runtime**: Node.js
- **Framework**: Express.js
- **Database**: MongoDB (Mongoose ODM)
- **Authentication**: JWT, Google OAuth 2.0
- **Security**: bcryptjs for password hashing

## Quick Start

### Prerequisites

- Node.js (v18 or higher)
- MongoDB (local or Atlas)
- npm or yarn

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd backend-pcg

# Install dependencies
npm install

# Create .env file
cp .env.example .env
# Edit .env with your configuration

# Start development server
npm run dev
```

### Environment Variables

Create a `.env` file in the root directory:

```env
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/dbname
JWT_SECRET=your-secret-token-here
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
PORT=5000
```

### Create Admin User

```bash
npm run create-admin
```

Follow the prompts to create an admin account.

## API Endpoints

### Authentication

- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `POST /api/auth/google` - Google OAuth login
- `GET /api/auth/me` - Get current user

### Tickets

- `GET /api/tickets` - Get all tickets
- `POST /api/tickets` - Create ticket
- `GET /api/tickets/:id` - Get ticket by ID
- `PUT /api/tickets/:id` - Update ticket
- `DELETE /api/tickets/:id` - Delete ticket

### Routing Rules

- `GET /api/routing-rules` - Get all routing rules
- `POST /api/routing-rules` - Create routing rule
- `PUT /api/routing-rules/:id` - Update routing rule
- `DELETE /api/routing-rules/:id` - Delete routing rule

## Development

```bash
# Start development server with auto-reload
npm run dev

# Start production server
npm start

# Create admin user
npm run create-admin
```

## Deployment

### Deploy to Render

See [RENDER_DEPLOYMENT.md](./RENDER_DEPLOYMENT.md) for detailed deployment instructions.

Quick deploy:
1. Push code to GitHub
2. Create new Web Service on [Render](https://render.com)
3. Connect your repository
4. Set environment variables
5. Deploy!

### Deploy to Other Platforms

- **Heroku**: Use `Procfile` with `web: npm start`
- **Railway**: Connect repo and deploy
- **DigitalOcean**: Use App Platform
- **AWS**: Use Elastic Beanstalk or EC2

## Project Structure

```
backend-pcg/
├── src/
│   ├── config/
│   │   └── db.js              # Database configuration
│   ├── models/
│   │   ├── User.js            # User model
│   │   ├── Ticket.js          # Ticket model
│   │   └── RoutingRule.js     # Routing rule model
│   ├── routes/
│   │   ├── authRoutes.js      # Authentication routes
│   │   ├── ticketRoutes.js    # Ticket routes
│   │   └── routingRoutes.js   # Routing rules routes
│   ├── middleware/
│   │   └── auth.js            # Authentication middleware
│   └── server.js              # Server entry point
├── scripts/
│   └── createAdminUser.js     # Admin user creation script
├── app.js                     # Express app configuration
├── package.json
├── .env                       # Environment variables (not in git)
├── .gitignore
└── README.md
```

## Security

- Passwords hashed with bcryptjs
- JWT tokens for authentication
- Environment variables for secrets
- CORS enabled for frontend integration
- Input validation and sanitization

## Testing

```bash
# Run tests (when available)
npm test
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License

ISC

## Support

For issues and questions:
- Create an issue in the repository
- Contact the development team

## Related Projects

- [Frontend (Next.js)](../pcg_frontend)
- [ML API (Python/FastAPI)](../itsm-ai-api)

## Changelog

See [CHANGELOG.md](./CHANGELOG.md) for version history.
