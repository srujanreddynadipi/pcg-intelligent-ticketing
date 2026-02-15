"use client";

import { useState } from "react";
import { apiClient } from "@/lib/api";
import { API_ENDPOINTS } from "@/lib/config";
import {
  transformBackendTicket,
  transformBackendTickets,
  transformTicket,
} from "@/lib/utils";
import { Ticket, CreateTicketPayload } from "@/lib/types";

export const useTickets = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const getAllTickets = async (): Promise<Ticket[]> => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await apiClient.get<Ticket[]>(
        API_ENDPOINTS.TICKETS.GET_ALL,
      );

      // Handle wrapped ApiResponse
      if (response.success && response.data && Array.isArray(response.data)) {
        return transformBackendTickets(response.data as any[]);
      }
      // Handle direct array response
      else if (Array.isArray(response)) {
        return transformBackendTickets(response as any[]);
      } else {
        setError("Failed to fetch tickets");
        return [];
      }
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Failed to fetch tickets";
      setError(errorMessage);
      return [];
    } finally {
      setIsLoading(false);
    }
  };

  const getTicketById = async (id: string): Promise<Ticket | null> => {
    setIsLoading(true);
    setError(null);
    try {
      const endpoint = API_ENDPOINTS.TICKETS.GET_BY_ID.replace(":id", id);
      const response = await apiClient.get<Ticket>(endpoint);

      // Handle wrapped ApiResponse
      if (response.success && response.data) {
        return transformBackendTicket(response.data as any);
      }
      // Handle direct object response
      else if (
        response &&
        typeof response === "object" &&
        "title" in response
      ) {
        return transformBackendTicket(response as any);
      } else {
        setError("Failed to fetch ticket");
        return null;
      }
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Failed to fetch ticket";
      setError(errorMessage);
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  const createTicket = async (
    payload: CreateTicketPayload & {
      category?: string;
      categoryConfidence?: number;
      mlPrediction?: any;
    },
  ): Promise<Ticket | null> => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await apiClient.post<Ticket>(
        API_ENDPOINTS.TICKETS.CREATE,
        payload,
      );

      // Handle wrapped ApiResponse
      if (response.success && response.data) {
        return transformBackendTicket(response.data as any);
      }
      // Handle direct object response
      else if (
        response &&
        typeof response === "object" &&
        "title" in response
      ) {
        return transformBackendTicket(response as any);
      } else {
        setError("Failed to create ticket");
        return null;
      }
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Failed to create ticket";
      setError(errorMessage);
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  const updateTicket = async (
    id: string,
    updates: Partial<Ticket>,
  ): Promise<Ticket | null> => {
    setIsLoading(true);
    setError(null);
    try {
      const endpoint = API_ENDPOINTS.TICKETS.UPDATE.replace(":id", id);
      const response = await apiClient.put<Ticket>(endpoint, updates);

      // Handle wrapped ApiResponse
      if (response.success && response.data) {
        return transformBackendTicket(response.data as any);
      }
      // Handle direct object response
      else if (
        response &&
        typeof response === "object" &&
        "title" in response
      ) {
        return transformBackendTicket(response as any);
      } else {
        setError("Failed to update ticket");
        return null;
      }
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Failed to update ticket";
      setError(errorMessage);
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  const deleteTicket = async (id: string): Promise<boolean> => {
    setIsLoading(true);
    setError(null);
    try {
      const endpoint = API_ENDPOINTS.TICKETS.DELETE.replace(":id", id);
      const response = await apiClient.delete(endpoint);
      if (response.success) {
        return true;
      } else {
        setError(response.error || "Failed to delete ticket");
        return false;
      }
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Failed to delete ticket";
      setError(errorMessage);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const getTicketsByStatus = async (
    status: "pending" | "resolved",
  ): Promise<Ticket[]> => {
    setIsLoading(true);
    setError(null);
    try {
      const endpoint = API_ENDPOINTS.TICKETS.GET_BY_STATUS.replace(
        ":status",
        status,
      );
      const response = await apiClient.get<Ticket[]>(endpoint);

      // Handle wrapped ApiResponse
      if (response.success && response.data && Array.isArray(response.data)) {
        return transformBackendTickets(response.data as any[]);
      }
      // Handle direct array response
      else if (Array.isArray(response)) {
        return transformBackendTickets(response as any[]);
      } else {
        setError("Failed to fetch tickets");
        return [];
      }
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Failed to fetch tickets";
      setError(errorMessage);
      return [];
    } finally {
      setIsLoading(false);
    }
  };

  const getTicketsByUserId = async (userId: string): Promise<Ticket[]> => {
    setIsLoading(true);
    setError(null);
    try {
      const endpoint = API_ENDPOINTS.TICKETS.GET_BY_USER.replace(
        ":userId",
        userId,
      );
      const response = await apiClient.get<Ticket[]>(endpoint);

      // Handle wrapped ApiResponse
      if (response.success && response.data && Array.isArray(response.data)) {
        return transformBackendTickets(response.data as any[]);
      }
      // Handle direct array response
      else if (Array.isArray(response)) {
        return transformBackendTickets(response as any[]);
      } else {
        setError("Failed to fetch user tickets");
        return [];
      }
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Failed to fetch user tickets";
      setError(errorMessage);
      return [];
    } finally {
      setIsLoading(false);
    }
  };

  return {
    isLoading,
    error,
    getAllTickets,
    getTicketById,
    createTicket,
    updateTicket,
    deleteTicket,
    getTicketsByStatus,
    getTicketsByUserId,
  };
};
