export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:3000";
export const ML_API_URL =
  process.env.NEXT_PUBLIC_ML_API_URL || "http://localhost:8000";
export const GOOGLE_CLIENT_ID = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID || "";

export const API_ENDPOINTS = {
  AUTH: {
    LOGIN: "/auth/login",
    SIGNUP: "/auth/signup",
    GOOGLE_LOGIN: "/auth/google-login",
    LOGOUT: "/auth/logout",
    PROFILE: "/auth/profile",
  },
  TICKETS: {
    CREATE: "/api/tickets",
    GET_ALL: "/api/tickets",
    GET_BY_ID: "/api/tickets/:id",
    UPDATE: "/api/tickets/:id",
    DELETE: "/api/tickets/:id",
    GET_BY_STATUS: "/api/tickets/status/:status",
    GET_BY_USER: "/api/tickets/user/:userId",
  },
  ML: {
    PREDICT: "/predict",
  },
} as const;
