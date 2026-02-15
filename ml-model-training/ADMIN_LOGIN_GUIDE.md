# Admin Login Setup Guide

## Admin Credentials

**Email:** `admin@gmail.com`  
**Password:** `admin123`

## How to Set Up Admin User

### Option 1: Run the Admin User Creation Script (Recommended)

1. Make sure your MongoDB is running
2. Navigate to the backend directory:
   ```bash
   cd C:\Users\sruja\Classroom\SysntheticDataPCG\backend-pcg
   ```
3. Run the script:
   ```bash
   npm run create-admin
   ```

This will create the admin user if it doesn't exist, or update the password if it does.

### Option 2: Sign Up with Admin Credentials

1. Start the backend server:
   ```bash
   cd C:\Users\sruja\Classroom\SysntheticDataPCG\backend-pcg
   npm start
   ```

2. Go to the signup page:
   ```
   http://localhost:3000/signup
   ```

3. Create an account with:
   - **Email:** admin@gmail.com
   - **Password:** admin123
   - **Name:** Admin

## Login to Admin Dashboard

### Direct Admin Login
Visit: `http://localhost:3000/admin/login`
- Email field is pre-filled with admin@gmail.com
- Enter password: admin123

### Regular Login (Auto-Redirect)
Visit: `http://localhost:3000/login`
- Enter email: admin@gmail.com
- Enter password: admin123
- Will automatically redirect to admin dashboard

## Features

✅ **Auto-Detection**: Login system automatically detects admin credentials and redirects to `/admin`

✅ **Dedicated Admin Login Page**: Accessible at `/admin/login` with pre-filled email

✅ **Protected Admin Routes**: All `/admin/*` routes require authentication

✅ **JWT Token Authentication**: Secure token-based authentication for API requests

## Troubleshooting

### "Invalid Credentials" Error
- Make sure MongoDB is running
- Run `npm run create-admin` to create/update the admin user
- Check backend console for error messages

### "Unauthorized" API Errors
- Check that JWT_SECRET is set in your `.env` file
- Verify the token is being stored (check browser console)
- Restart the backend server after making changes

### Backend Not Starting
- Check MongoDB connection in `.env`:
  ```
  MONGO_URI=mongodb://localhost:27017/your-database-name
  JWT_SECRET=your-secret-key-here
  ```
- Make sure all dependencies are installed: `npm install`
