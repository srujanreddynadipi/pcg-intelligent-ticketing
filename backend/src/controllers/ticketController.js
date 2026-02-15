const Ticket = require('../models/Ticket');

// @desc    Create a new ticket with ML classification
// @route   POST /api/tickets
// @access  Private
exports.createTicket = async (req, res) => {
    const { title, description, category, priority, resolver_group, historical_tickets, knowledge_base, ml_classification } = req.body;

    try {
        console.log('[Ticket] Auth user ID:', req.user.id);
        console.log('[Ticket] Creating new ticket:', { title, category, priority });
        console.log('[Ticket] Full request body:', req.body);
        
        // Validate required fields
        if (!title || !description) {
            return res.status(400).json({ msg: 'Title and description are required' });
        }

        if (!req.user || !req.user.id) {
            return res.status(401).json({ msg: 'User not authenticated' });
        }

        const newTicket = new Ticket({
            user: req.user.id,
            email: req.body.email || req.body.user,
            title,
            description,
            category: category || 'General',
            priority: priority || 'Low',
            resolver_group,
            historical_tickets: historical_tickets || [],
            knowledge_base: knowledge_base || [],
            ml_classification: ml_classification ? {
                ...ml_classification,
                classified_at: new Date()
            } : null
        });

        console.log('[Ticket] Ticket object:', newTicket);
        const ticket = await newTicket.save();
        console.log('[Ticket] Ticket created successfully:', ticket._id);
        res.json(ticket);
    } catch (err) {
        console.error('[Ticket] Create error:', err);
        console.error('[Ticket] Error stack:', err.stack);
        res.status(500).json({ 
            msg: 'Server Error', 
            error: err.message,
            details: err.errors ? Object.values(err.errors).map(e => e.message) : undefined
        });
    }
};

// @desc    Analyze ticket with ML classification
// @route   POST /api/tickets/analyze
// @access  Private
exports.analyzeTicket = async (req, res) => {
    const { title, description, historical_tickets, knowledge_base } = req.body;

    try {
        console.log('[Analyze] Processing ticket analysis:', { title });
        
        // TODO: Call your ML model here
        // For now, returning the structure that was sent
        const mlResponse = {
            detail: {
                error: "Analysis completed",
                message: "Ticket has been analyzed and classified",
                recommendations: [
                    "Review ticket classification",
                    "Route to appropriate department",
                    "Follow up with user if needed"
                ],
                ticket_id: `TICKET-${Date.now()}`
            }
        };

        console.log('[Analyze] ML response:', mlResponse);
        res.json(mlResponse);
    } catch (err) {
        console.error('[Analyze] Analysis error:', err.message);
        res.status(500).json({ msg: 'Analysis failed', error: err.message });
    }
};

// @desc    Get all tickets for logged in user
// @route   GET /api/tickets
// @access  Private
exports.getTickets = async (req, res) => {
    try {
        const tickets = await Ticket.find().sort({ created_at: -1 });
        console.log('[Tickets] Retrieved', tickets.length, 'tickets');
        res.json(tickets);
    } catch (err) {
        console.error('[Tickets] Get error:', err.message);
        res.status(500).json({ msg: 'Server Error', error: err.message });
    }
};

// @desc    Get user's tickets with ML classification
// @route   GET /api/tickets/user/:userId
// @access  Private
exports.getUserTickets = async (req, res) => {
    try {
        const userId = req.params.userId || req.user.id;
        const tickets = await Ticket.find({ user: userId }).sort({ created_at: -1 });
        console.log('[UserTickets] Retrieved', tickets.length, 'tickets for user:', userId);
        res.json(tickets);
    } catch (err) {
        console.error('[UserTickets] Get error:', err.message);
        res.status(500).json({ msg: 'Server Error', error: err.message });
    }
};

// @desc    Get ticket by ID
// @route   GET /api/tickets/:id
// @access  Private
exports.getTicketById = async (req, res) => {
    try {
        const ticket = await Ticket.findById(req.params.id);

        if (!ticket) {
            return res.status(404).json({ msg: 'Ticket not found' });
        }

        // Make sure user owns ticket
        if (ticket.user.toString() !== req.user.id) {
            return res.status(401).json({ msg: 'Not authorized' });
        }

        console.log('[TicketById] Retrieved ticket:', ticket._id);
        res.json(ticket);
    } catch (err) {
        console.error('[TicketById] Get error:', err.message);
        if (err.kind === 'ObjectId') {
            return res.status(404).json({ msg: 'Ticket not found' });
        }
        res.status(500).json({ msg: 'Server Error', error: err.message });
    }
};

// @desc    Update ticket
// @route   PUT /api/tickets/:id
// @access  Public (no authentication required)
exports.updateTicket = async (req, res) => {
    const { title, description, category, priority, status, resolver_group, ml_classification } = req.body;

    // Build ticket object
    const ticketFields = {};
    if (title) ticketFields.title = title;
    if (description) ticketFields.description = description;
    if (category) ticketFields.category = category;
    if (priority) ticketFields.priority = priority;
    if (status) {
        ticketFields.status = status;
        if (status === 'closed') {
            ticketFields.resolved_at = Date.now();
        }
    }
    if (resolver_group) ticketFields.resolver_group = resolver_group;
    if (ml_classification) {
        ticketFields.ml_classification = {
            ...ml_classification,
            classified_at: new Date()
        };
    }

    try {
        let ticket = await Ticket.findById(req.params.id);

        if (!ticket) return res.status(404).json({ msg: 'Ticket not found' });
        // Make sure user owns ticket

        ticket = await Ticket.findByIdAndUpdate(
            req.params.id,
            { $set: ticketFields },
            { new: true }
        );

        console.log('[Update] Ticket updated:', ticket._id);
        res.json(ticket);
    } catch (err) {
        console.error('[Update] Error:', err.message);
        res.status(500).json({ msg: 'Server Error', error: err.message });
    }
};

// @desc    Delete ticket
// @route   DELETE /api/tickets/:id
// @access  Private
exports.deleteTicket = async (req, res) => {
    try {
        const ticket = await Ticket.findById(req.params.id);

        if (!ticket) return res.status(404).json({ msg: 'Ticket not found' });

        // Make sure user owns ticket
        if (ticket.user.toString() !== req.user.id) {
            return res.status(401).json({ msg: 'Not authorized' });
        }

        await Ticket.findByIdAndDelete(req.params.id);

        console.log('[Delete] Ticket deleted:', req.params.id);
        res.json({ msg: 'Ticket removed' });
    } catch (err) {
        console.error('[Delete] Error:', err.message);
        res.status(500).json({ msg: 'Server Error', error: err.message });
    }
};
