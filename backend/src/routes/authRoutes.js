const express = require('express');
const router = express.Router();
const authController = require('../controllers/authController');
const auth = require('../middleware/authMiddleware');
const User = require('../models/User');

// @route   POST api/auth/signup
// @desc    Register user
// @access  Public
router.post('/signup', authController.signup);

// @route   POST api/auth/login
// @desc    Authenticate user & get token
// @access  Public
router.post('/login', authController.login);

// @route   POST api/auth/google
// @desc    Google authentication
// @access  Public
router.post('/google', authController.googleAuth);

// @route   POST api/auth/google-login
// @desc    Google login (alternative endpoint)
// @access  Public
router.post('/google-login', authController.googleAuth);

// @route   GET api/auth/user
// @desc    Get user data
// @access  Private
router.get('/user', auth, async (req, res) => {
    try {
        const user = await User.findById(req.user.id).select('-password');
        res.json(user);
    } catch (err) {
        console.error(err.message);
        res.status(500).json({ msg: 'Server Error' });
    }
});

// @route   GET api/auth/profile
// @desc    Get user profile
// @access  Private
router.get('/profile', auth, async (req, res) => {
    try {
        console.log('[Profile] Fetching profile for user:', req.user.id);
        const user = await User.findById(req.user.id).select('-password');
        if (!user) {
            return res.status(404).json({ 
                success: false, 
                message: 'User not found',
                error: 'User not found' 
            });
        }
        console.log('[Profile] User found:', user.email);
        
        // Return in frontend-expected format
        res.json({ 
            success: true,
            data: {
                id: user._id || user.id,
                email: user.email,
                name: user.firstName || user.email.split('@')[0],
                userId: user.userId,
                googleId: user.googleId,
                createdAt: user.createdAt,
                updatedAt: user.updatedAt
            },
            message: 'Profile fetched successfully'
        });
    } catch (err) {
        console.error('[Profile] Error:', err.message);
        res.status(500).json({ 
            success: false, 
            message: 'Server Error', 
            error: err.message 
        });
    }
});

// @route   POST api/auth/logout
// @desc    Logout user (client-side token deletion)
// @access  Public
router.post('/logout', (req, res) => {
    res.json({ 
        success: true, 
        message: 'Logged out successfully' 
    });
});

module.exports = router;
