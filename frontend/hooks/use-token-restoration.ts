import { useEffect } from 'react';
import { apiClient } from '@/lib/api';

/**
 * Hook to ensure JWT token is restored from localStorage
 * This prevents API calls from failing after page refresh
 */
export const useTokenRestoration = () => {
  useEffect(() => {
    const token = localStorage.getItem('authToken');
    if (token && !apiClient.getToken()) {
      console.log('[TokenRestoration] Restoring token from localStorage');
      apiClient.setToken(token);
    }
  }, []);
};
