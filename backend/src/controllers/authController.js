const User = require('../models/User');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const mongoose = require('mongoose');
const { OAuth2Client } = require('google-auth-library');
const client = new OAuth2Client(process.env.GOOGLE_CLIENT_ID);

exports.signup = async (req, res) => {
    const { email, password, firstName } = req.body;

    try {
        let user = await User.findOne({ email });

        if (user) {
            return res.status(400).json({ 
                success: false, 
                message: 'User already exists',
                error: 'User already exists' 
            });
        }

        user = new User({
            userId: new mongoose.Types.ObjectId().toString(),
            email,
            password,
            firstName
        });

        const salt = await bcrypt.genSalt(10);
        user.password = await bcrypt.hash(password, salt);

        await user.save();

        const payload = {
            user: {
                id: user.id
            }
        };

        jwt.sign(
            payload,
            process.env.JWT_SECRET,
            { expiresIn: 360000 },
            (err, token) => {
                if (err) {
                    console.error('JWT Sign Error:', err);
                    return res.status(500).json({ 
                        success: false, 
                        message: 'Token generation failed',
                        error: err.message 
                    });
                }
                
                // Return in frontend-expected format
                res.json({ 
                    success: true,
                    data: {
                        token,
                        user: {
                            id: user._id || user.id,
                            email: user.email,
                            name: user.firstName || user.email.split('@')[0],
                            userId: user.userId
                        }
                    },
                    message: 'Signup successful'
                });
            }
        );
    } catch (err) {
        console.error('Signup Error:', err);
        res.status(500).json({ 
            success: false, 
            message: 'Server error', 
            error: err.message 
        });
    }
};

exports.login = async (req, res) => {
    const { email, password } = req.body;

    try {
        let user = await User.findOne({ email });

        if (!user) {
            return res.status(400).json({ 
                success: false, 
                message: 'Invalid Credentials',
                error: 'Invalid Credentials' 
            });
        }

        const isMatch = await bcrypt.compare(password, user.password);

        if (!isMatch) {
            return res.status(400).json({ 
                success: false, 
                message: 'Invalid Credentials',
                error: 'Invalid Credentials' 
            });
        }

        const payload = {
            user: {
                id: user.id
            }
        };

        jwt.sign(
            payload,
            process.env.JWT_SECRET,
            { expiresIn: 360000 },
            (err, token) => {
                if (err) {
                    console.error('JWT Sign Error:', err);
                    return res.status(500).json({ 
                        success: false, 
                        message: 'Token generation failed',
                        error: err.message 
                    });
                }
                
                // Return in frontend-expected format
                res.json({ 
                    success: true,
                    data: {
                        token,
                        user: {
                            id: user._id || user.id,
                            email: user.email,
                            name: user.firstName || user.email.split('@')[0],
                            userId: user.userId
                        }
                    },
                    message: 'Login successful'
                });
            }
        );
    } catch (err) {
        console.error('Login Error:', err);
        res.status(500).json({ 
            success: false, 
            message: 'Server error', 
            error: err.message 
        });
    }
};

exports.googleAuth = async (req, res) => {
    const { token, idToken } = req.body;
    const googleToken = token || idToken;

    console.log('Google Auth Request Body:', req.body);
    console.log('Token received:', !!googleToken);

    if (!googleToken) {
        return res.status(400).json({ 
            success: false, 
            message: 'No token provided', 
            error: 'No token provided',
            received: req.body 
        });
    }

    try {
        console.log('Verifying ID Token...');
        const ticket = await client.verifyIdToken({
            idToken: googleToken,
            audience: process.env.GOOGLE_CLIENT_ID
        });

        console.log('Token verified successfully');
        const { email, sub, given_name } = ticket.getPayload();
        console.log('Extracted payload:', { email, sub, given_name });

        let user = await User.findOne({ email });

        if (!user) {
            console.log('Creating new user...');
            user = new User({
                userId: sub,
                email,
                googleId: sub,
                firstName: given_name
            });
            await user.save();
            console.log('User created:', user._id);
        } else {
            console.log('User found:', user._id);
        }

        const payload = {
            user: {
                id: user.id
            }
        };

        jwt.sign(
            payload,
            process.env.JWT_SECRET,
            { expiresIn: 360000 },
            (err, token) => {
                if (err) {
                    console.error('JWT Sign Error:', err);
                    return res.status(500).json({ 
                        success: false, 
                        message: 'Token generation failed', 
                        error: err.message 
                    });
                }
                console.log('Token generated successfully');
                
                // Return in frontend-expected format
                res.json({ 
                    success: true,
                    data: {
                        token,
                        user: {
                            id: user._id || user.id,
                            email: user.email,
                            name: user.firstName || user.email.split('@')[0],
                            userId: user.userId,
                            googleId: user.googleId
                        }
                    },
                    message: 'Google login successful'
                });
            }
        );
    } catch (err) {
        console.error('Google Auth Error:', err);
        res.status(500).json({ 
            success: false, 
            message: 'Google login failed', 
            error: err.message 
        });
    }
};
