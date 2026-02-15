const mongoose = require('mongoose');

const RoutingRuleSchema = new mongoose.Schema({
    name: {
        type: String,
        required: true
    },
    category: {
        type: String,
        required: true,
        enum: ['Network', 'Hardware', 'Software', 'Access', 'Database', 'Security', 'Cloud', 'DevOps', 'Email', 'Monitoring', 'Service Request']
    },
    priority: {
        type: String,
        required: true,
        enum: ['Low', 'Medium', 'High', 'Critical']
    },
    keywords: {
        type: [String],
        default: []
    },
    resolver_group: {
        type: String,
        required: true
    },
    confidence_threshold: {
        type: Number,
        default: 75,
        min: 0,
        max: 100
    },
    active: {
        type: Boolean,
        default: true
    },
    created_at: {
        type: Date,
        default: Date.now
    },
    updated_at: {
        type: Date,
        default: Date.now
    }
});

// Update the updated_at timestamp before saving
RoutingRuleSchema.pre('save', function(next) {
    this.updated_at = Date.now();
    next();
});

module.exports = mongoose.model('RoutingRule', RoutingRuleSchema);
