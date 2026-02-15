const express = require('express');
const router = express.Router();
const routingController = require('../controllers/routingController');
const auth = require('../middleware/authMiddleware');

// All routes are protected
router.use(auth);

// Routing rules CRUD
router.get('/', routingController.getAllRules);
router.get('/stats', routingController.getCategoryStats);
router.get('/category/:category', routingController.getRulesByCategory);
router.get('/:id', routingController.getRuleById);
router.post('/', routingController.createRule);
router.put('/:id', routingController.updateRule);
router.delete('/:id', routingController.deleteRule);

module.exports = router;
