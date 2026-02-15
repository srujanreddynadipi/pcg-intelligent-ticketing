"use client";

import { useAuth } from "@/lib/auth-context";
import { useTokenRestoration } from "@/hooks/use-token-restoration";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import Link from "next/link";
import { apiClient } from "@/lib/api";
import { API_ENDPOINTS } from "@/lib/config";
import { transformBackendTickets } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Ticket } from "@/lib/types";

export default function DashboardPage() {
  const { isAuthenticated, isLoading, user } = useAuth();
  useTokenRestoration();
  const router = useRouter();
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [ticketsLoading, setTicketsLoading] = useState(true);

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push("/login");
    }
  }, [isAuthenticated, isLoading, router]);

  // Fetch user's tickets
  useEffect(() => {
    if (!isAuthenticated || !user?.id) return;

    const fetchTickets = async () => {
      setTicketsLoading(true);
      try {
        const endpoint = API_ENDPOINTS.TICKETS.GET_BY_USER.replace(
          ":userId",
          user.id,
        );

        const response = await apiClient.get<Ticket[]>(endpoint);

        // Handle case where backend returns tickets wrapped in ApiResponse
        if (response.success && response.data) {
          const transformedTickets = transformBackendTickets(response.data);
          setTickets(transformedTickets);
        }
        // Handle case where backend returns plain array
        else if (Array.isArray(response)) {
          const transformedTickets = transformBackendTickets(response as any[]);
          setTickets(transformedTickets);
        }
        // Handle case where data is directly in response object as array
        else if (response && typeof response === "object") {
          const dataValue = (response as any).data;
          if (Array.isArray(dataValue)) {
            const transformedTickets = transformBackendTickets(dataValue);
            setTickets(transformedTickets);
          }
        }
      } catch (err) {
        console.error("[Dashboard] Fetch error:", err);
      } finally {
        setTicketsLoading(false);
      }
    };

    fetchTickets();
  }, [isAuthenticated, user?.id]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin">
          <div className="w-12 h-12 border-4 border-slate-700 border-t-blue-600 rounded-full" />
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return null;
  }

  // Calculate stats
  const totalTickets = tickets.length;
  const pendingCount = tickets.filter(
    (t) => t.status?.toLowerCase() === "pending",
  ).length;
  const resolvedCount = tickets.filter(
    (t) => t.status?.toLowerCase() === "resolved",
  ).length;

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Welcome Section */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-white mb-2">
            Welcome back, {user?.name.split(" ")[0]}!
          </h1>
          <p className="text-slate-400 text-lg">
            Manage your tickets and track issues efficiently
          </p>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <Card className="bg-slate-800 border-slate-700 p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm mb-1">Total Tickets</p>
                <p className="text-3xl font-bold text-white">{totalTickets}</p>
              </div>
              <div className="w-12 h-12 bg-blue-600 bg-opacity-20 rounded-lg flex items-center justify-center">
                <svg
                  className="w-6 h-6 text-blue-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
                  />
                </svg>
              </div>
            </div>
          </Card>

          <Card className="bg-slate-800 border-slate-700 p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm mb-1">Pending</p>
                <p className="text-3xl font-bold text-white">{pendingCount}</p>
              </div>
              <div className="w-12 h-12 bg-yellow-600 bg-opacity-20 rounded-lg flex items-center justify-center">
                <svg
                  className="w-6 h-6 text-yellow-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
            </div>
          </Card>

          <Card className="bg-slate-800 border-slate-700 p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm mb-1">Resolved</p>
                <p className="text-3xl font-bold text-white">{resolvedCount}</p>
              </div>
              <div className="w-12 h-12 bg-green-600 bg-opacity-20 rounded-lg flex items-center justify-center">
                <svg
                  className="w-6 h-6 text-green-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
            </div>
          </Card>
        </div>

        {/* Action Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card className="bg-gradient-to-br from-blue-900 to-blue-800 border-blue-700 p-8">
            <div className="mb-4">
              <div className="w-12 h-12 bg-blue-500 rounded-lg flex items-center justify-center mb-4">
                <svg
                  className="w-6 h-6 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 4v16m8-8H4"
                  />
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-white mb-2">
                Create New Ticket
              </h3>
              <p className="text-blue-200 mb-6">
                Report an issue and let our AI categorize it automatically
              </p>
            </div>
            <Link href="/tickets/create">
              <Button className="bg-white text-blue-600 hover:bg-blue-50 font-semibold w-full">
                Create Ticket
              </Button>
            </Link>
          </Card>

          <Card className="bg-gradient-to-br from-slate-700 to-slate-600 border-slate-600 p-8">
            <div className="mb-4">
              <div className="w-12 h-12 bg-slate-500 rounded-lg flex items-center justify-center mb-4">
                <svg
                  className="w-6 h-6 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
                  />
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-white mb-2">
                View All Tickets
              </h3>
              <p className="text-slate-300 mb-6">
                Browse, filter, and manage all your support tickets
              </p>
            </div>
            <Link href="/tickets">
              <Button className="bg-slate-500 hover:bg-slate-400 text-white font-semibold w-full">
                View Tickets
              </Button>
            </Link>
          </Card>
        </div>
      </div>
    </main>
  );
}
