const express = require('express');
const router = express.Router();
const ticketController = require('../controllers/ticketController');
// REMOVED AUTH - ALL ROUTES ARE NOW PUBLIC

// Analyze endpoint (for ML classification)
router.post('/analyze', ticketController.analyzeTicket);

// Ticket CRUD endpoints - ALL PUBLIC
router.post('/', ticketController.createTicket);
router.post('/create', ticketController.createTicket); // Alternative endpoint for frontend compatibility
router.get('/', ticketController.getTickets);
router.get('/user/:userId', ticketController.getUserTickets);
router.get('/:id', ticketController.getTicketById);
router.put('/:id', ticketController.updateTicket);
router.delete('/:id', ticketController.deleteTicket);

module.exports = router;
