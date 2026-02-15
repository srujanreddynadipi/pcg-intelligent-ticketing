'use client';

import React, { createContext, useContext, useEffect, useState } from 'react';
import { User, AuthResponse } from './types';
import { apiClient } from './api';
import { API_ENDPOINTS, API_BASE_URL } from './config';

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string, name: string) => Promise<void>;
  googleLogin: (idToken: string) => Promise<void>;
  logout: () => Promise<void>;
  error: string | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Check if user is logged in on mount
  useEffect(() => {
    const checkAuth = async () => {
      console.log('[Auth] Checking auth on mount...');
      const token = localStorage.getItem('authToken');
      const cachedUser = localStorage.getItem('user');
      
      if (token) {
        console.log('[Auth] Token found in localStorage, restoring...');
        apiClient.setToken(token);
        
        // If cached user exists, use it immediately
        if (cachedUser) {
          try {
            const parsedUser = JSON.parse(cachedUser);
            console.log('[Auth] Restored user from cache:', parsedUser);
            setUser(parsedUser);
          } catch (e) {
            console.warn('[Auth] Failed to parse cached user');
          }
        }
        
        // Always verify token with backend profile
        try {
          const response = await apiClient.get<User>(API_ENDPOINTS.AUTH.PROFILE);
          console.log('[Auth] Profile response:', response);
          
          if (response.success && response.data) {
            console.log('[Auth] Profile verified, setting user:', response.data);
            setUser(response.data);
            localStorage.setItem('user', JSON.stringify(response.data));
          } else if (response.data && typeof response.data === 'object' && 'email' in response.data) {
            // Handle raw format
            console.log('[Auth] Profile returned in raw format');
            const userData = response.data as User;
            setUser(userData);
            localStorage.setItem('user', JSON.stringify(userData));
          } else {
            console.warn('[Auth] Invalid profile response, clearing auth');
            localStorage.removeItem('authToken');
            localStorage.removeItem('user');
            apiClient.clearToken();
          }
        } catch (err) {
          console.error('[Auth] Profile verification failed:', err);
          localStorage.removeItem('authToken');
          localStorage.removeItem('user');
          apiClient.clearToken();
        }
      } else {
        console.log('[Auth] No token found');
        localStorage.removeItem('user');
      }
      setIsLoading(false);
    };

    checkAuth();
  }, []);

  const login = async (email: string, password: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await apiClient.post<AuthResponse>(
        API_ENDPOINTS.AUTH.LOGIN,
        { email, password }
      );

      if (response.success && response.data) {
        console.log('[Auth] Login successful');
        apiClient.setToken(response.data.token);
        // Persist both token and user
        localStorage.setItem('authToken', response.data.token);
        localStorage.setItem('user', JSON.stringify(response.data.user));
        setUser(response.data.user);
      } else {
        setError(response.error || 'Login failed');
        throw new Error(response.error || 'Login failed');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Login failed';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const signup = async (email: string, password: string, name: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await apiClient.post<AuthResponse>(
        API_ENDPOINTS.AUTH.SIGNUP,
        { email, password, name }
      );

      if (response.success && response.data) {
        console.log('[Auth] Signup successful');
        apiClient.setToken(response.data.token);
        // Persist both token and user
        localStorage.setItem('authToken', response.data.token);
        localStorage.setItem('user', JSON.stringify(response.data.user));
        setUser(response.data.user);
      } else {
        setError(response.error || 'Signup failed');
        throw new Error(response.error || 'Signup failed');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Signup failed';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const googleLogin = async (idToken: string) => {
    setIsLoading(true);
    setError(null);
    try {
      console.log('[Auth] Sending Google login request to:', API_ENDPOINTS.AUTH.GOOGLE_LOGIN);
      const rawResponse = await fetch(`${API_BASE_URL}${API_ENDPOINTS.AUTH.GOOGLE_LOGIN}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ idToken }),
      });
      
      const responseData = await rawResponse.json();
      console.log('[Auth] Google login raw response:', responseData);

      // Handle both formats: ApiResponse and backend's raw format
      let token: string | null = null;
      let user: User | null = null;

      if (responseData.success && responseData.data) {
        // Standard ApiResponse format
        token = responseData.data.token;
        user = responseData.data.user;
      } else if (responseData.token && responseData.userId) {
        // Backend's raw format - need to fetch user profile using the token
        token = responseData.token as string;
        console.log('[Auth] Got token from raw response, fetching user profile...');
        
        // Set token temporarily to fetch profile
        if (token) {
          apiClient.setToken(token);
        }
        try {
          // Directly fetch to see raw response
          const profileRawResponse = await fetch(`${API_BASE_URL}${API_ENDPOINTS.AUTH.PROFILE}`, {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${token || ''}`,
            },
          });
          
          console.log('[Auth] Profile response status:', profileRawResponse.status);
          const profileData = await profileRawResponse.json();
          console.log('[Auth] Profile raw response data:', profileData);

          if (profileRawResponse.ok && profileData.success && profileData.data) {
            user = profileData.data;
          } else if (profileRawResponse.ok && profileData.data && typeof profileData.data === 'object' && 'email' in profileData.data) {
            // Handle case where backend returns user object directly in data
            user = profileData.data as User;
          } else if (profileRawResponse.ok && typeof profileData === 'object' && '_id' in profileData && 'email' in profileData) {
            // Handle backend's raw format with _id, firstName, etc.
            user = {
              id: profileData._id,
              email: profileData.email,
              name: `${profileData.firstName || ''} ${profileData.lastName || ''}`.trim(),
              avatar: profileData.avatar,
              googleId: profileData.googleId,
              createdAt: profileData.createdAt,
              updatedAt: profileData.updatedAt,
            } as User;
          } else if (profileRawResponse.ok && typeof profileData === 'object' && 'email' in profileData && 'id' in profileData) {
            // Handle case where backend returns user object directly
            user = profileData as User;
          } else {
            console.error('[Auth] Unexpected profile response format:', profileData);
            throw new Error('Invalid profile response format');
          }
        } catch (profileErr) {
          console.error('[Auth] Profile fetch error:', profileErr);
          apiClient.clearToken();
          throw profileErr;
        }
      } else {
        throw new Error('Invalid response format from server');
      }

      if (token && user) {
        console.log('[Auth] Response successful, setting user:', user);
        apiClient.setToken(token);
        // Store both token and user in localStorage
        localStorage.setItem('authToken', token);
        localStorage.setItem('user', JSON.stringify(user));
        setUser(user);
      } else {
        throw new Error('Missing token or user data');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Google login failed';
      console.error('[Auth] Google login exception:', errorMessage, err);
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = async () => {
    setIsLoading(true);
    try {
      await apiClient.post(API_ENDPOINTS.AUTH.LOGOUT, {});
      apiClient.clearToken();
      setUser(null);
      // Clear both token and user from localStorage
      localStorage.removeItem('authToken');
      localStorage.removeItem('user');
      console.log('[Auth] Logged out successfully');
    } catch (err) {
      console.error('Logout failed:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        login,
        signup,
        googleLogin,
        logout,
        error,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};
