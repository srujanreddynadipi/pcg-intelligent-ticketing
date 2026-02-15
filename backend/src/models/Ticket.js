const mongoose = require('mongoose');

const TicketSchema = new mongoose.Schema({
    ticket_id: {
        type: String,
        required: true,
        unique: true,
        default: () => `TICKET-${Date.now()}`
    },
    user: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    email: {
        type: String
    },
    title: {
        type: String,
        required: true
    },
    description: {
        type: String,
        required: true
    },
    category: {
        type: String
    },
    priority: {
        type: String,
        default: 'Low'
    },
    status: {
        type: String,
        enum: ['active', 'pending', 'closed'],
        default: 'active'
    },
    resolver_group: {
        type: String
    },
    // ML Classification fields
    ml_classification: {
        error: String,
        message: String,
        recommendations: [String],
        is_it_ticket: Boolean,
        classified_at: Date
    },
    // Historical tickets
    historical_tickets: [{
        ticket_id: String,
        title: String,
        description: String,
        status: String,
        resolution: String
    }],
    // Knowledge base articles
    knowledge_base: [{
        article_id: String,
        title: String,
        solution: String,
        category: String
    }],
    created_at: {
        type: Date,
        default: Date.now
    },
    resolved_at: {
        type: Date
    }
});

module.exports = mongoose.model('Ticket', TicketSchema);
