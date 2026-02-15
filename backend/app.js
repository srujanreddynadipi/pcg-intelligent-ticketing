const express = require('express');
const cors = require('cors');
const connectDB = require('./src/config/db');

const app = express();

// Init Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Debug Middleware to inspect request
app.use((req, res, next) => {
    console.log(`[DEBUG] ${req.method} ${req.url}`);
    console.log('[DEBUG] Content-Type:', req.get('Content-Type'));
    if (!req.body) {
        console.warn('[WARNING] req.body is undefined. Initializing to empty object.');
        req.body = {};
    }
    next();
});

// Connect Database
connectDB();

// Define Routes
app.use('/api/auth', require('./src/routes/authRoutes'));
app.use('/api/tickets', require('./src/routes/ticketRoutes'));
app.use('/api/routing-rules', require('./src/routes/routingRoutes'));

// Alternative routes without /api prefix (for client compatibility)
app.use('/auth', require('./src/routes/authRoutes'));
app.use('/tickets', require('./src/routes/ticketRoutes'));
app.use('/routing-rules', require('./src/routes/routingRoutes'));

module.exports = app;
