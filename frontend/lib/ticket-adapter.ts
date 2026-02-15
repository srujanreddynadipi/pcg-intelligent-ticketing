// c:\Users\sruja\Classroom\SysntheticDataPCG\pcg_frontend\lib\ticket-adapter.ts

import { Ticket, ApiResponse } from './types';

/**
 * Raw ticket structure from backend MongoDB
 */
export interface RawTicket {
  _id?: string;
  id?: string;
  ticket_id?: string;
  title?: string;
  description?: string;
  status?: string;
  priority?: string;
  category?: string;
  category_confidence?: number;
  categoryConfidence?: number;
  assignee?: string;
  assigned_to?: string;
  resolver_group?: string;
  created_by?: string;
  createdBy?: string;
  created_at?: string;
  createdAt?: string;
  updated_at?: string;
  updatedAt?: string;
  user?: string;
  email?: string;
  ml_classification?: any;
  mlPrediction?: any;
  historical_tickets?: any[];
  knowledge_base?: any[];
}

/**
 * Normalize status values from backend to frontend format
 */
export const normalizeStatus = (status?: string): 'pending' | 'resolved' => {
  if (!status) return 'pending';
  const lower = status.toLowerCase();
  
  // Map backend status values to frontend
  if (lower === 'closed' || lower === 'resolved') {
    return 'resolved';
  }
  return 'pending'; // 'active', 'pending', 'open', or any other value
};

/**
 * Normalize priority values from backend to frontend format
 */
export const normalizePriority = (priority?: string): 'low' | 'medium' | 'high' => {
  if (!priority) return 'medium';
  const lower = priority.toLowerCase();
  
  // Map backend priority values to frontend
  if (lower === 'critical' || lower === 'urgent') {
    return 'high';
  }
  if (lower === 'high') {
    return 'high';
  }
  if (lower === 'low') {
    return 'low';
  }
  return 'medium'; // default
};

/**
 * Map a single raw ticket from backend to frontend Ticket type
 */
export const mapTicket = (raw: RawTicket): Ticket => {
  return {
    id: String(raw?.id || raw?._id || raw?.ticket_id || ''),
    title: raw?.title || 'Untitled',
    description: raw?.description || '',
    status: normalizeStatus(raw?.status),
    priority: normalizePriority(raw?.priority),
    category: raw?.category || 'Uncategorized',
    categoryConfidence: raw?.category_confidence || raw?.categoryConfidence || 0,
    assignee: raw?.assignee || raw?.assigned_to,
    resolver_group: raw?.resolver_group,
    createdBy: raw?.created_by || raw?.createdBy || raw?.user || 'Unknown',
    createdAt: raw?.created_at || raw?.createdAt || new Date().toISOString(),
    updatedAt: raw?.updated_at || raw?.updatedAt || new Date().toISOString(),
    user: raw?.user,
    email: raw?.email,
    ml_classification: raw?.ml_classification,
    mlPrediction: raw?.mlPrediction,
    historical_tickets: raw?.historical_tickets,
    knowledge_base: raw?.knowledge_base,
  };
};

/**
 * Map tickets from backend response (handles both array and ApiResponse formats)
 */
export const mapTickets = (responseOrArray: any): Ticket[] => {
  // Handle null/undefined
  if (!responseOrArray) {
    return [];
  }

  // If it's already a Ticket array, return as-is
  if (Array.isArray(responseOrArray) && responseOrArray.length > 0) {
    // Check if first item looks like a frontend Ticket (has 'id' not '_id')
    const first = responseOrArray[0];
    if (first.id && !first._id) {
      return responseOrArray as Ticket[];
    }
    // Otherwise map it
    return responseOrArray.map(mapTicket);
  }

  // Handle ApiResponse wrapper
  if (responseOrArray.success && responseOrArray.data) {
    if (Array.isArray(responseOrArray.data)) {
      return responseOrArray.data.map(mapTicket);
    }
    // Single ticket wrapped in ApiResponse
    return [mapTicket(responseOrArray.data)];
  }

  // Handle raw array
  if (Array.isArray(responseOrArray)) {
    return responseOrArray.map(mapTicket);
  }

  // Single ticket object
  return [mapTicket(responseOrArray)];
};

/**
 * Helper to normalize a single ticket (for ticket detail pages)
 */
export const normalizeTicket = (raw: any): Ticket => {
  return mapTicket(raw);
};