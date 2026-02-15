import { API_BASE_URL, ML_API_URL } from "./config";
import { ApiResponse } from "./types";

export class ApiClient {
  private baseUrl: string;
  private token: string | null = null;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
    // Get token from localStorage on client side
    if (typeof window !== "undefined") {
      this.token = localStorage.getItem("authToken");
    }
  }

  setToken(token: string) {
    this.token = token;
    if (typeof window !== "undefined") {
      localStorage.setItem("authToken", token);
    }
  }

  getToken() {
    return this.token;
  }

  clearToken() {
    this.token = null;
    if (typeof window !== "undefined") {
      localStorage.removeItem("authToken");
    }
  }

  private getHeaders(): HeadersInit {
    const headers: HeadersInit = {
      "Content-Type": "application/json",
    };

    if (this.token) {
      headers["Authorization"] = `Bearer ${this.token}`;
    }

    return headers;
  }

  async get<T>(endpoint: string): Promise<ApiResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: "GET",
        headers: this.getHeaders(),
      });

      // Check if response is OK before parsing JSON
      if (!response.ok) {
        const contentType = response.headers.get("content-type");
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`;

        // Try to parse error response if it's JSON
        if (contentType?.includes("application/json")) {
          try {
            const errorData = await response.json();
            errorMessage = errorData.message || errorData.error || errorMessage;
          } catch {
            // If JSON parse fails, use the status message
          }
        }

        return {
          success: false,
          message: "Failed to fetch data",
          error: errorMessage,
        };
      }

      // Parse successful response
      const data = await response.json();
      return data as ApiResponse<T>;
    } catch (error) {
      return {
        success: false,
        message: "Failed to fetch data",
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }

  async post<T>(endpoint: string, data: unknown): Promise<ApiResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: "POST",
        headers: this.getHeaders(),
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        const contentType = response.headers.get("content-type");
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`;

        if (contentType?.includes("application/json")) {
          try {
            const errorData = await response.json();
            errorMessage = errorData.message || errorData.error || errorMessage;
          } catch {
            // If JSON parse fails, use the status message
          }
        }

        return {
          success: false,
          message: "Failed to post data",
          error: errorMessage,
        };
      }

      const result = await response.json();
      // Ensure success is set to true for 2xx responses
      return {
        ...result,
        success: result.success !== false,
      };
    } catch (error) {
      return {
        success: false,
        message: "Failed to post data",
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }

  async put<T>(endpoint: string, data: unknown): Promise<ApiResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: "PUT",
        headers: this.getHeaders(),
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        const contentType = response.headers.get("content-type");
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`;

        if (contentType?.includes("application/json")) {
          try {
            const errorData = await response.json();
            errorMessage = errorData.message || errorData.error || errorMessage;
          } catch {
            // If JSON parse fails, use the status message
          }
        }

        return {
          success: false,
          message: "Failed to update data",
          error: errorMessage,
        };
      }

      const result = await response.json();
      // Ensure success is set to true for 2xx responses
      return {
        ...result,
        success: result.success !== false,
      };
    } catch (error) {
      return {
        success: false,
        message: "Failed to update data",
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }

  async delete<T>(endpoint: string): Promise<ApiResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: "DELETE",
        headers: this.getHeaders(),
      });

      return await response.json();
    } catch (error) {
      return {
        success: false,
        message: "Failed to delete data",
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }
}

export class MLApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = ML_API_URL) {
    this.baseUrl = baseUrl;
  }

  async predict<T>(data: unknown): Promise<T> {
    try {
      const response = await fetch(`${this.baseUrl}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        throw new Error(`ML API error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error("ML prediction failed:", error);
      throw error;
    }
  }
}

export const apiClient = new ApiClient();
export const mlApiClient = new MLApiClient();
