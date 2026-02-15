# TicketHub - AI-Powered Ticket Management System

A production-ready ticket management system frontend built with Next.js, featuring AI-powered ticket categorization via machine learning integration.

## Features

✨ **Core Features**
- User authentication (Email/Password & Google OAuth)
- Create, read, update, and delete tickets
- AI-powered automatic ticket categorization
- Real-time confidence scores for predictions
- Filter tickets by status and priority
- Search functionality across all tickets
- Notification center
- Responsive dark theme design

🔐 **Security**
- JWT token-based authentication
- Protected routes and endpoints
- Secure token storage
- Input validation
- CORS protection

🚀 **Performance**
- Next.js 16 with App Router
- Client-side filtering and search
- Optimized component rendering
- Responsive design for all devices

## Quick Start

### Prerequisites
- Node.js 18+ (16+ recommended)
- pnpm (or npm)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd tickethub-frontend

# Install dependencies
pnpm install

# Create environment file
echo "NEXT_PUBLIC_API_URL=http://localhost:3000
NEXT_PUBLIC_ML_API_URL=http://localhost:8000
NEXT_PUBLIC_GOOGLE_CLIENT_ID=your_google_client_id" > .env.local

# Start development server
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
tickethub-frontend/
├── app/
│   ├── page.tsx                 # Landing page
│   ├── login/page.tsx           # Login page
│   ├── signup/page.tsx          # Signup page
│   ├── dashboard/page.tsx       # Dashboard
│   ├── tickets/                 # Ticket pages
│   │   ├── page.tsx
│   │   ├── create/page.tsx
│   │   └── [id]/page.tsx
│   ├── notifications/page.tsx   # Notifications
│   ├── layout.tsx               # Root layout
│   └── globals.css              # Global styles
├── components/
│   ├── auth/                    # Auth components
│   ├── layout/                  # Layout components
│   └── ui/                      # UI components
├── lib/
│   ├── config.ts                # Configuration
│   ├── types.ts                 # TypeScript interfaces
│   ├── api.ts                   # API clients
│   └── auth-context.tsx         # Auth provider
├── hooks/
│   └── use-tickets.ts           # Tickets hook
└── Documentation files
```

## Documentation

Detailed documentation is available in the following files:

- **[QUICK_START.md](./QUICK_START.md)** - Quick setup and common tasks
- **[IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md)** - Complete setup and development guide
- **[BACKEND_INTEGRATION.md](./BACKEND_INTEGRATION.md)** - Backend API specifications
- **[API_REFERENCE.md](./API_REFERENCE.md)** - Detailed API endpoint reference
- **[PROJECT_SUMMARY.md](./PROJECT_SUMMARY.md)** - Project overview and deliverables

## Environment Variables

Required environment variables in `.env.local`:

```env
# Backend API
NEXT_PUBLIC_API_URL=http://localhost:3000

# ML Service
NEXT_PUBLIC_ML_API_URL=http://localhost:8000

# Google OAuth
NEXT_PUBLIC_GOOGLE_CLIENT_ID=your_google_client_id_here
```

## Backend Requirements

The frontend expects these services:

1. **Backend API** on port 3000
   - Authentication endpoints
   - Ticket management endpoints
   - JWT token support

2. **ML Service** on port 8000
   - `/predict` endpoint for text classification

See [BACKEND_INTEGRATION.md](./BACKEND_INTEGRATION.md) for detailed specifications.

## Technology Stack

- **Framework**: Next.js 16
- **React**: 19.2.3
- **Language**: TypeScript
- **Styling**: Tailwind CSS + shadcn/ui
- **State Management**: React Context + Hooks
- **API**: Fetch API with custom client
- **Authentication**: JWT + Google OAuth

## Available Scripts

```bash
# Development
pnpm dev              # Start development server
pnpm build            # Build for production
pnpm start            # Start production server
pnpm lint             # Run linter

# Alternative with npm
npm run dev
npm run build
npm start
npm run lint
```

## Pages & Routes

| Route | Purpose | Protected |
|-------|---------|-----------|
| `/` | Landing page | No |
| `/login` | Login | No |
| `/signup` | Register | No |
| `/dashboard` | Dashboard | Yes |
| `/tickets` | All tickets | Yes |
| `/tickets/create` | Create ticket | Yes |
| `/tickets/:id` | Ticket detail | Yes |
| `/notifications` | Notifications | Yes |

## Key Components

### Authentication
- Email/password login and signup
- Google OAuth integration
- Protected routes
- Token management

### Ticket Management
- Create tickets with ML categorization
- List and filter tickets
- View detailed ticket information
- Update ticket status
- Search functionality

### ML Integration
- Real-time text categorization
- Confidence score display
- Alternative predictions
- Prediction storage

### Notifications
- Notification center
- Read/unread tracking
- Notification filtering
- Real-time updates

## Design

### Theme
- **Color Scheme**: Dark theme with blue accents
- **Primary Color**: Blue (#0066cc)
- **Neutral Colors**: Slate palette (800-950)
- **Accent Colors**: Green, Yellow, Red, Purple

### Responsive Design
- Mobile-first approach
- Breakpoints: sm (640px), md (768px), lg (1024px)
- Flexible grid layouts
- Touch-friendly components

## Development

### Adding New Pages

Create a new file in `app/` directory:

```typescript
'use client';

import { useAuth } from '@/lib/auth-context';

export default function NewPage() {
  const { isAuthenticated, isLoading } = useAuth();

  if (!isAuthenticated) return null;

  return <main>{/* Content */}</main>;
}
```

### Making API Calls

```typescript
import { apiClient, API_ENDPOINTS } from '@/lib/config';

const response = await apiClient.get(API_ENDPOINTS.TICKETS.GET_ALL);
```

### Using Authentication

```typescript
import { useAuth } from '@/lib/auth-context';

const { user, login, logout, isAuthenticated } = useAuth();
```

## Deployment

### Deploy to Vercel

```bash
# Connect your GitHub repository to Vercel
# Set environment variables in Vercel dashboard
# Push to GitHub - automatic deployment
```

### Deploy Elsewhere

```bash
# Build the application
pnpm build

# Start production server
pnpm start
```

## Troubleshooting

### Issues & Solutions

**"Cannot GET /"**
- Backend not running on port 3000
- Check `NEXT_PUBLIC_API_URL` environment variable

**"ML predictions failing"**
- ML service not running on port 8000
- Check `NEXT_PUBLIC_ML_API_URL` environment variable

**"Login not working"**
- Backend authentication endpoints not implemented
- Check JWT token generation
- Verify CORS configuration

**"Blank page"**
- Check browser console for errors
- Verify environment variables are set
- Ensure all services are running

## Support

For issues or questions:

1. Check the documentation files
2. Review the browser console for errors
3. Verify backend and ML services are running
4. Check environment variable configuration

## Testing

### Manual Testing Checklist

- [ ] Create an account (signup)
- [ ] Login with credentials
- [ ] Login with Google OAuth
- [ ] Create a ticket
- [ ] Verify ML predictions
- [ ] View all tickets
- [ ] Filter tickets by status
- [ ] Filter tickets by priority
- [ ] Search tickets
- [ ] View ticket details
- [ ] Update ticket status
- [ ] View notifications
- [ ] Logout

## Performance

- Optimized bundle size
- Client-side filtering for instant search
- Lazy component loading
- Efficient re-rendering
- Responsive images

## Security

- JWT token authentication
- Secure token storage (localStorage)
- Protected routes
- Input validation
- HTTPS recommended
- CORS configured
- XSS protection via React

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)
- Mobile browsers

## Contributing

Guidelines for contributing to this project:

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request
5. Code review and merge

## License

[Your License Here]

## Changelog

### Version 1.0.0 (Current)
- Initial release
- Complete ticket management system
- AI-powered categorization
- User authentication
- Responsive design

## Roadmap

Future enhancements:
- Advanced analytics
- Team collaboration
- Ticket templates
- Custom workflows
- Mobile app
- Real-time collaboration
- Webhook integrations

## Contact & Support

For support, documentation, or questions:
- Check documentation files
- Review API reference
- Refer to backend integration guide

---

**Status**: Production Ready ✅  
**Last Updated**: 2024  
**Version**: 1.0.0

Built with ❤️ using Next.js and TypeScript
