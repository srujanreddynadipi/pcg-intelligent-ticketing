require('dotenv').config();
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const User = require('../src/models/User');

const ADMIN_EMAIL = 'admin@gmail.com';
const ADMIN_PASSWORD = 'admin123';
const ADMIN_NAME = 'Admin';

async function createAdminUser() {
    try {
        // Connect to MongoDB
        await mongoose.connect(process.env.MONGO_URI);
        console.log('✅ Connected to MongoDB');

        // Check if admin already exists
        let admin = await User.findOne({ email: ADMIN_EMAIL });
        
        if (admin) {
            console.log('ℹ️  Admin user already exists:', ADMIN_EMAIL);
            
            // Update password in case it changed
            const salt = await bcrypt.genSalt(10);
            admin.password = await bcrypt.hash(ADMIN_PASSWORD, salt);
            await admin.save();
            console.log('✅ Admin password updated');
        } else {
            // Create new admin user
            admin = new User({
                userId: new mongoose.Types.ObjectId().toString(),
                email: ADMIN_EMAIL,
                password: ADMIN_PASSWORD,
                firstName: ADMIN_NAME,
                role: 'admin'
            });

            // Hash password
            const salt = await bcrypt.genSalt(10);
            admin.password = await bcrypt.hash(ADMIN_PASSWORD, salt);

            // Save to database
            await admin.save();
            console.log('✅ Admin user created successfully!');
        }

        console.log('\n📋 Admin Credentials:');
        console.log('   Email:', ADMIN_EMAIL);
        console.log('   Password:', ADMIN_PASSWORD);
        console.log('   URL: http://localhost:3000/admin/login');
        
        process.exit(0);
    } catch (error) {
        console.error('❌ Error creating admin user:', error.message);
        process.exit(1);
    }
}

createAdminUser();
