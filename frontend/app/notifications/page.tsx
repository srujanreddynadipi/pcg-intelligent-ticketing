'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/lib/auth-context';
import { useTokenRestoration } from '@/hooks/use-token-restoration';
import { apiClient } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Notification } from '@/lib/types';

export default function NotificationsPage() {
  const { isAuthenticated } = useAuth();
  useTokenRestoration();
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [filter, setFilter] = useState<'all' | 'unread'>('all');

  useEffect(() => {
    if (!isAuthenticated) return;

    const fetchNotifications = async () => {
      setIsLoading(true);
      setError('');
      try {
        // Ensure token is set before any API call
        const token = localStorage.getItem('authToken');
        if (token) {
          apiClient.setToken(token);
        }

        // For now, using mock notifications
        // TODO: Replace with actual API call when backend endpoint is ready
        console.log('[Notifications] Loading notifications...');
        
        const mockNotifications: Notification[] = [
          {
            id: '1',
            userId: 'user-1',
            title: 'Ticket Created',
            message: 'Your ticket "Login page bug" has been created successfully',
            type: 'ticket_created',
            ticketId: 'ticket-1',
            read: false,
            createdAt: new Date(Date.now() - 3600000).toISOString(),
          },
          {
            id: '2',
            userId: 'user-1',
            title: 'Ticket Updated',
            message: 'Ticket "Payment gateway error" has been updated',
            type: 'ticket_updated',
            ticketId: 'ticket-2',
            read: false,
            createdAt: new Date(Date.now() - 7200000).toISOString(),
          },
          {
            id: '3',
            userId: 'user-1',
            title: 'Ticket Resolved',
            message: 'Ticket "Dashboard loading" has been resolved',
            type: 'ticket_resolved',
            ticketId: 'ticket-3',
            read: true,
            createdAt: new Date(Date.now() - 86400000).toISOString(),
          },
          {
            id: '4',
            userId: 'user-1',
            title: 'Ticket Created',
            message: 'Your ticket "API timeout issue" has been created successfully',
            type: 'ticket_created',
            ticketId: 'ticket-4',
            read: true,
            createdAt: new Date(Date.now() - 172800000).toISOString(),
          },
        ];
        
        console.log('[Notifications] Loaded', mockNotifications.length, 'notifications');
        setNotifications(mockNotifications);
      } catch (err) {
        const error = err instanceof Error ? err.message : 'Failed to load notifications';
        console.error('[Notifications] Fetch error:', error);
        setError(error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchNotifications();
  }, [isAuthenticated]);

  if (!isAuthenticated) {
    return null;
  }

  const filteredNotifications =
    filter === 'unread' ? notifications.filter((n) => !n.read) : notifications;

  const unreadCount = notifications.filter((n) => !n.read).length;

  const markAsRead = (id: string) => {
    setNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, read: true } : n))
    );
  };

  const markAllAsRead = () => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
  };

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'ticket_created':
        return (
          <div className="w-10 h-10 bg-blue-500 bg-opacity-20 rounded-full flex items-center justify-center">
            <svg className="w-6 h-6 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 4v16m8-8H4"
              />
            </svg>
          </div>
        );
      case 'ticket_updated':
        return (
          <div className="w-10 h-10 bg-yellow-500 bg-opacity-20 rounded-full flex items-center justify-center">
            <svg className="w-6 h-6 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
              />
            </svg>
          </div>
        );
      case 'ticket_resolved':
        return (
          <div className="w-10 h-10 bg-green-500 bg-opacity-20 rounded-full flex items-center justify-center">
            <svg className="w-6 h-6 text-green-400" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                clipRule="evenodd"
              />
            </svg>
          </div>
        );
      default:
        return (
          <div className="w-10 h-10 bg-slate-500 bg-opacity-20 rounded-full flex items-center justify-center">
            <svg className="w-6 h-6 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
              />
            </svg>
          </div>
        );
    }
  };

  const formatTime = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2">Notifications</h1>
            <p className="text-slate-400">Stay updated on your tickets</p>
          </div>
          {unreadCount > 0 && (
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-400">{unreadCount}</div>
              <p className="text-slate-400 text-sm">Unread</p>
            </div>
          )}
        </div>

        {/* Filter and Actions */}
        <Card className="bg-slate-800 border-slate-700 p-4 mb-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex gap-2">
            <Button
              onClick={() => setFilter('all')}
              className={`${
                filter === 'all'
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-700 hover:bg-slate-600 text-slate-200'
              } font-medium`}
            >
              All
            </Button>
            <Button
              onClick={() => setFilter('unread')}
              className={`${
                filter === 'unread'
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-700 hover:bg-slate-600 text-slate-200'
              } font-medium`}
            >
              Unread ({unreadCount})
            </Button>
          </div>
          {unreadCount > 0 && (
            <Button
              onClick={markAllAsRead}
              className="bg-slate-700 hover:bg-slate-600 text-slate-200 font-medium"
            >
              Mark all as read
            </Button>
          )}
        </Card>

        {/* Loading State */}
        {isLoading && (
          <div className="flex justify-center py-12">
            <div className="animate-spin">
              <div className="w-12 h-12 border-4 border-slate-700 border-t-blue-600 rounded-full" />
            </div>
          </div>
        )}

        {/* Error State */}
        {error && (
          <Card className="bg-red-500 bg-opacity-10 border border-red-500 p-4 mb-8">
            <p className="text-red-400">{error}</p>
          </Card>
        )}

        {/* Notifications List */}
        {!isLoading && (
          <div className="space-y-4">
            {filteredNotifications.length > 0 ? (
              filteredNotifications.map((notification) => (
                <Card
                  key={notification.id}
                  className={`border-slate-700 p-6 transition-colors ${
                    notification.read
                      ? 'bg-slate-800'
                      : 'bg-slate-750 border-l-4 border-l-blue-600'
                  }`}
                >
                  <div className="flex items-start gap-4">
                    {getNotificationIcon(notification.type)}
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-white mb-1">
                        {notification.title}
                      </h3>
                      <p className="text-slate-300 text-sm mb-2">{notification.message}</p>
                      <p className="text-slate-500 text-xs">{formatTime(notification.createdAt)}</p>
                    </div>
                    {!notification.read && (
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-blue-600 rounded-full" />
                        <Button
                          onClick={() => markAsRead(notification.id)}
                          className="text-xs bg-slate-700 hover:bg-slate-600 text-slate-200 px-3 py-1 h-auto"
                        >
                          Read
                        </Button>
                      </div>
                    )}
                  </div>
                </Card>
              ))
            ) : (
              <Card className="bg-slate-800 border-slate-700 p-12 text-center">
                <svg
                  className="w-12 h-12 text-slate-500 mx-auto mb-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
                  />
                </svg>
                <p className="text-slate-400 text-lg">No notifications</p>
                <p className="text-slate-500 text-sm mt-1">You're all caught up!</p>
              </Card>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
