import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { Ticket } from "./types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Normalizes priority casing to lowercase
 */
function normalizePriority(priority: string): "low" | "medium" | "high" {
  const normalized = priority.toLowerCase();
  if (
    normalized === "low" ||
    normalized === "medium" ||
    normalized === "high"
  ) {
    return normalized;
  }
  return "low"; // default fallback
}

/**
 * Maps backend status values to frontend status values
 * Backend: "active" or "closed"
 * Frontend: "pending" (for active) or "resolved" (for closed)
 */
function mapBackendStatus(status: string): "pending" | "resolved" {
  if (!status) return "pending";

  const lowerStatus = status.toLowerCase().trim();

  // Map backend "closed" status to frontend "resolved"
  if (lowerStatus === "closed") {
    return "resolved";
  }

  // Map any other status ("active", etc) to "pending"
  return "pending";
}

/**
 * Transforms backend response to frontend Ticket format
 */
export function transformBackendTicket(backendTicket: any): Ticket {
  return {
    id: backendTicket._id || backendTicket.id,
    title: backendTicket.title,
    description: backendTicket.description,
    status: mapBackendStatus(backendTicket.status),
    priority: normalizePriority(backendTicket.priority),
    category: backendTicket.category || "Uncategorized",
    categoryConfidence: backendTicket.ml_classification ? 1 : 0,
    createdAt: backendTicket.created_at || backendTicket.createdAt,
    user: backendTicket.user,
    email: backendTicket.email,
    ticket_id: backendTicket.ticket_id,
    resolver_group: backendTicket.resolver_group,
    historical_tickets: backendTicket.historical_tickets || [],
    knowledge_base: backendTicket.knowledge_base || [],
    ml_classification: backendTicket.ml_classification,
    // Mark as closed if status is either 'pending' or 'resolved'
    closed: true,
  };
}

/**
 * Transforms an array of backend tickets to frontend format
 */
export function transformBackendTickets(backendTickets: any[]): Ticket[] {
  return backendTickets.map(transformBackendTicket);
}

/**
 * Transforms frontend ticket data from create/update operations
 * Classifies both 'pending' and 'resolved' statuses as 'closed'
 */
export function transformTicket(ticket: Ticket): Ticket {
  return {
    ...ticket,
    // Mark as closed if status is either 'pending' or 'resolved'
    closed: ticket.status === "pending" || ticket.status === "resolved",
  };
}

/**
 * Transforms an array of tickets from frontend operations
 */
export function transformTickets(tickets: Ticket[]): Ticket[] {
  return tickets.map(transformTicket);
}
