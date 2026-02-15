const RoutingRule = require('../models/RoutingRule');

// Get all routing rules
exports.getAllRules = async (req, res) => {
    try {
        const rules = await RoutingRule.find().sort({ created_at: -1 });
        res.json(rules);
    } catch (error) {
        console.error('Error fetching routing rules:', error);
        res.status(500).json({ 
            success: false,
            message: 'Error fetching routing rules',
            error: error.message 
        });
    }
};

// Get routing rules by category
exports.getRulesByCategory = async (req, res) => {
    try {
        const { category } = req.params;
        const rules = await RoutingRule.find({ category }).sort({ created_at: -1 });
        res.json(rules);
    } catch (error) {
        console.error('Error fetching rules by category:', error);
        res.status(500).json({ 
            success: false,
            message: 'Error fetching rules by category',
            error: error.message 
        });
    }
};

// Get single routing rule by ID
exports.getRuleById = async (req, res) => {
    try {
        const rule = await RoutingRule.findById(req.params.id);
        if (!rule) {
            return res.status(404).json({ 
                success: false,
                message: 'Routing rule not found' 
            });
        }
        res.json(rule);
    } catch (error) {
        console.error('Error fetching routing rule:', error);
        res.status(500).json({ 
            success: false,
            message: 'Error fetching routing rule',
            error: error.message 
        });
    }
};

// Create new routing rule
exports.createRule = async (req, res) => {
    try {
        const { name, category, priority, keywords, resolver_group, confidence_threshold, active } = req.body;

        // Validation
        if (!name || !category || !priority || !resolver_group) {
            return res.status(400).json({ 
                success: false,
                message: 'Name, category, priority, and resolver group are required' 
            });
        }

        const rule = new RoutingRule({
            name,
            category,
            priority,
            keywords: keywords || [],
            resolver_group,
            confidence_threshold: confidence_threshold || 75,
            active: active !== undefined ? active : true
        });

        await rule.save();
        
        res.status(201).json({
            success: true,
            message: 'Routing rule created successfully',
            data: rule
        });
    } catch (error) {
        console.error('Error creating routing rule:', error);
        res.status(500).json({ 
            success: false,
            message: 'Error creating routing rule',
            error: error.message 
        });
    }
};

// Update routing rule
exports.updateRule = async (req, res) => {
    try {
        const { id } = req.params;
        const updates = req.body;

        const rule = await RoutingRule.findByIdAndUpdate(
            id,
            { ...updates, updated_at: Date.now() },
            { new: true, runValidators: true }
        );

        if (!rule) {
            return res.status(404).json({ 
                success: false,
                message: 'Routing rule not found' 
            });
        }

        res.json({
            success: true,
            message: 'Routing rule updated successfully',
            data: rule
        });
    } catch (error) {
        console.error('Error updating routing rule:', error);
        res.status(500).json({ 
            success: false,
            message: 'Error updating routing rule',
            error: error.message 
        });
    }
};

// Delete routing rule
exports.deleteRule = async (req, res) => {
    try {
        const { id } = req.params;
        const rule = await RoutingRule.findByIdAndDelete(id);

        if (!rule) {
            return res.status(404).json({ 
                success: false,
                message: 'Routing rule not found' 
            });
        }

        res.json({
            success: true,
            message: 'Routing rule deleted successfully'
        });
    } catch (error) {
        console.error('Error deleting routing rule:', error);
        res.status(500).json({ 
            success: false,
            message: 'Error deleting routing rule',
            error: error.message 
        });
    }
};

// Get category statistics
exports.getCategoryStats = async (req, res) => {
    try {
        const stats = await RoutingRule.aggregate([
            {
                $group: {
                    _id: '$category',
                    count: { $sum: 1 },
                    activeCount: {
                        $sum: { $cond: ['$active', 1, 0] }
                    }
                }
            },
            {
                $project: {
                    category: '$_id',
                    count: 1,
                    activeCount: 1,
                    _id: 0
                }
            },
            {
                $sort: { count: -1 }
            }
        ]);

        res.json(stats);
    } catch (error) {
        console.error('Error fetching category stats:', error);
        res.status(500).json({ 
            success: false,
            message: 'Error fetching category statistics',
            error: error.message 
        });
    }
};
