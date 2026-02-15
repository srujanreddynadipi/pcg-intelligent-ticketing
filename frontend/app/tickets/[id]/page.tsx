'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth-context';
import { useTokenRestoration } from '@/hooks/use-token-restoration';
import { apiClient } from '@/lib/api';
import { API_ENDPOINTS } from '@/lib/config';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Ticket } from '@/lib/types';

export default function TicketDetailPage() {
  const params = useParams();
  const router = useRouter();
  const { isAuthenticated } = useAuth();
  useTokenRestoration();
  const [ticket, setTicket] = useState<Ticket | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [updatingStatus, setUpdatingStatus] = useState(false);

  const ticketId = params.id as string;

  useEffect(() => {
    if (!isAuthenticated || !ticketId) return;

    const fetchTicket = async () => {
      setIsLoading(true);
      setError('');
      try {
        // Ensure token is set before API call
        const token = localStorage.getItem('authToken');
        if (token) {
          apiClient.setToken(token);
        }

        const endpoint = API_ENDPOINTS.TICKETS.GET_BY_ID.replace(':id', ticketId);
        console.log('[TicketDetail] Fetching ticket from:', endpoint);
        const response = await apiClient.get<Ticket>(endpoint);

        console.log('[TicketDetail] API Response:', response);

        if (response.success && response.data) {
          console.log('[TicketDetail] Ticket loaded:', response.data);
          setTicket(response.data);
        } else if (response && typeof response === 'object' && 'title' in response) {
          // Handle case where backend returns ticket directly
          console.log('[TicketDetail] Ticket loaded (direct format)');
          if ('data' in response) {
  setTicket(response.data as Ticket);
}

        } else if (response && typeof response === 'object' && (response as any).data && 'title' in (response as any).data) {
          // Handle case where ticket is in data field
          console.log('[TicketDetail] Ticket loaded (in data field)');
          setTicket((response as any).data as Ticket);
        } else {
          console.error('[TicketDetail] Response error:', (response as any).error);
          setError((response as any).error || 'Failed to load ticket');
        }
      } catch (err) {
        const error = err instanceof Error ? err.message : 'Failed to load ticket';
        console.error('[TicketDetail] Fetch error:', error);
        setError(error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchTicket();
  }, [isAuthenticated, ticketId]);

  if (!isAuthenticated) {
    return null;
  }

  const handleStatusChange = async (newStatus: 'pending' | 'resolved') => {
    if (!ticket) return;

    setUpdatingStatus(true);
    try {
      // Ensure token is set before API call
      const token = localStorage.getItem('authToken');
      if (token) {
        apiClient.setToken(token);
      }

      const endpoint = API_ENDPOINTS.TICKETS.UPDATE.replace(':id', ticket.id);
      console.log('[TicketDetail] Updating ticket status to:', newStatus);
      const response = await apiClient.put(endpoint, { status: newStatus });

      console.log('[TicketDetail] Update response:', response);

      if (response.success || response.data) {
        console.log('[TicketDetail] Status updated successfully');
        setTicket({ ...ticket, status: newStatus });
      } else {
        console.error('[TicketDetail] Update error:', (response as any).error);
        setError((response as any).error || 'Failed to update ticket');
      }
    } catch (err) {
      const error = err instanceof Error ? err.message : 'Failed to update ticket';
      console.error('[TicketDetail] Update error:', error);
      setError(error);
    } finally {
      setUpdatingStatus(false);
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'bg-red-500 text-red-200';
      case 'medium':
        return 'bg-yellow-500 text-yellow-200';
      case 'low':
        return 'bg-green-500 text-green-200';
      default:
        return 'bg-slate-500 text-slate-200';
    }
  };

  const getPriorityBgColor = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'bg-red-500 bg-opacity-10 border-red-500';
      case 'medium':
        return 'bg-yellow-500 bg-opacity-10 border-yellow-500';
      case 'low':
        return 'bg-green-500 bg-opacity-10 border-green-500';
      default:
        return 'bg-slate-500 bg-opacity-10 border-slate-500';
    }
  };

  if (isLoading) {
    return (
      <main className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="flex justify-center py-12">
            <div className="animate-spin">
              <div className="w-12 h-12 border-4 border-slate-700 border-t-blue-600 rounded-full" />
            </div>
          </div>
        </div>
      </main>
    );
  }

  if (!ticket) {
    return (
      <main className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <Card className="bg-slate-800 border-slate-700 p-8 text-center">
            <p className="text-slate-400 text-lg mb-4">Ticket not found</p>
            <Button
              onClick={() => router.push('/tickets')}
              className="bg-blue-600 hover:bg-blue-700 text-white font-medium"
            >
              Back to Tickets
            </Button>
          </Card>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="mb-8">
          <Button
            onClick={() => router.push('/tickets')}
            className="mb-4 bg-slate-700 hover:bg-slate-600 text-slate-200"
          >
            ← Back to Tickets
          </Button>
          <h1 className="text-4xl font-bold text-white">{ticket.title}</h1>
        </div>

        {error && (
          <Card className="bg-red-500 bg-opacity-10 border border-red-500 p-4 mb-8">
            <p className="text-red-400">{error}</p>
          </Card>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            {/* Description */}
            <Card className="bg-slate-800 border-slate-700 p-8">
              <h2 className="text-xl font-semibold text-white mb-4">Description</h2>
              <p className="text-slate-300 leading-relaxed whitespace-pre-wrap">
                {ticket.description}
              </p>
            </Card>

            {/* AI Prediction */}
            {ticket.mlPrediction && (
              <Card className="bg-slate-800 border-slate-700 p-8">
                <h2 className="text-xl font-semibold text-white mb-4">AI Analysis</h2>
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-slate-300">Predicted Category:</span>
                      <span className="text-lg font-bold text-blue-400">
                        {ticket.mlPrediction.predicted_class}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-300">Confidence:</span>
                      <div className="flex items-center gap-2">
                        <div className="w-32 bg-slate-700 rounded-full h-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full"
                            style={{
                              width: `${ticket.categoryConfidence * 100}%`,
                            }}
                          />
                        </div>
                        <span className="text-slate-300">{(ticket.categoryConfidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>

                  {Object.keys(ticket.mlPrediction.confidence_scores).length > 1 && (
                    <div>
                      <p className="text-sm font-medium text-slate-300 mb-3">Other categories:</p>
                      <div className="space-y-2">
                        {Object.entries(ticket.mlPrediction.confidence_scores)
                          .filter(([category]) => category !== ticket.mlPrediction?.predicted_class)
                          .map(([category, score]) => (
                            <div key={category} className="flex items-center justify-between text-sm">
                              <span className="text-slate-400">{category}</span>
                              <span className="text-slate-300">{((score as number) * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}
                </div>
              </Card>
            )}

            {/* Activity Timeline */}
            <Card className="bg-slate-800 border-slate-700 p-8">
              <h2 className="text-xl font-semibold text-white mb-4">Timeline</h2>
              <div className="space-y-4">
                <div className="flex gap-4">
                  <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                    <svg
                      className="w-4 h-4 text-white"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </div>
                  <div>
                    <p className="font-medium text-white">Ticket Created</p>
                    <p className="text-slate-400 text-sm">
                      {new Date(ticket.createdAt).toLocaleString()}
                    </p>
                  </div>
                </div>
              </div>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Status */}
            <Card className="bg-slate-800 border-slate-700 p-6">
              <h3 className="text-sm font-semibold text-slate-200 mb-4">Status</h3>
              <div className="flex items-center gap-2 mb-4">
                <div
                  className={`w-3 h-3 rounded-full ${
                    ticket.status === 'resolved' ? 'bg-green-500' : 'bg-yellow-500'
                  }`}
                />
                <span className="text-white font-medium capitalize">{ticket.status}</span>
              </div>
              <div className="space-y-2">
                <Button
                  onClick={() => handleStatusChange('pending')}
                  disabled={updatingStatus || ticket.status === 'pending'}
                  className={`w-full ${
                    ticket.status === 'pending'
                      ? 'bg-yellow-600 text-white'
                      : 'bg-slate-700 hover:bg-slate-600 text-slate-200'
                  }`}
                >
                  Mark Pending
                </Button>
                <Button
                  onClick={() => handleStatusChange('resolved')}
                  disabled={updatingStatus || ticket.status === 'resolved'}
                  className={`w-full ${
                    ticket.status === 'resolved'
                      ? 'bg-green-600 text-white'
                      : 'bg-slate-700 hover:bg-slate-600 text-slate-200'
                  }`}
                >
                  Mark Resolved
                </Button>
              </div>
            </Card>

            {/* Priority */}
            <Card className="bg-slate-800 border-slate-700 p-6">
              <h3 className="text-sm font-semibold text-slate-200 mb-4">Priority</h3>
              <span
                className={`inline-block px-4 py-2 rounded-full font-medium capitalize ${getPriorityColor(
                  ticket.priority
                )}`}
              >
                {ticket.priority}
              </span>
            </Card>

            {/* Category */}
            <Card className="bg-slate-800 border-slate-700 p-6">
              <h3 className="text-sm font-semibold text-slate-200 mb-4">Category</h3>
              <div
                className={`border rounded-lg p-3 text-center ${getPriorityBgColor(
                  ticket.priority
                )}`}
              >
                <p className="text-sm font-medium text-slate-200">{ticket.category}</p>
                {ticket.categoryConfidence > 0 && (
                  <p className="text-xs text-slate-400 mt-1">
                    {(ticket.categoryConfidence * 100).toFixed(1)}% confidence
                  </p>
                )}
              </div>
            </Card>

            {/* Metadata */}
            <Card className="bg-slate-800 border-slate-700 p-6">
              <h3 className="text-sm font-semibold text-slate-200 mb-4">Details</h3>
              <div className="space-y-3 text-sm">
                <div>
                  <p className="text-slate-400">Created</p>
                  <p className="text-slate-200">
                    {new Date(ticket.createdAt).toLocaleDateString()}
                  </p>
                </div>
                <div>
                  <p className="text-slate-400">Last Updated</p>
                  <p className="text-slate-200">
                    {ticket.updatedAt && 
  new Date(ticket.updatedAt).toLocaleDateString()
}

                  </p>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </main>
  );
}
