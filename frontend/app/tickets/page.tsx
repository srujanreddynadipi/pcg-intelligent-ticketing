"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth-context";
import { useTokenRestoration } from "@/hooks/use-token-restoration";
import { apiClient } from "@/lib/api";
import { API_ENDPOINTS } from "@/lib/config";
import { transformBackendTickets, transformBackendTicket } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Ticket } from "@/lib/types";

type StatusFilter = "all" | "pending" | "resolved";
type PriorityFilter = "all" | "low" | "medium" | "high";

export default function TicketsPage() {
  const { isAuthenticated, user } = useAuth();
  useTokenRestoration();
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string>("");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [priorityFilter, setPriorityFilter] = useState<PriorityFilter>("all");
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    if (!isAuthenticated || !user?.id) return;

    const fetchTickets = async () => {
      setIsLoading(true);
      setError("");
      try {
        // Ensure token is set before API call
        const token = localStorage.getItem("authToken");
        if (token) {
          apiClient.setToken(token);
        }

        const endpoint = API_ENDPOINTS.TICKETS.GET_BY_USER.replace(
          ":userId",
          user.id,
        );
        console.log("[Tickets] Fetching user tickets from:", endpoint);
        const response = await apiClient.get<Ticket[]>(endpoint);

        console.log("[Tickets] API Response:", response);

        // Handle case where backend returns tickets wrapped in ApiResponse
        if (response.success && response.data) {
          console.log("[Tickets] Loaded", response.data.length, "tickets");
          // Data is already in frontend format
          const transformedTickets = transformBackendTickets(response.data);
          setTickets(transformedTickets);
        }
        // Handle case where backend returns plain array
        else if (Array.isArray(response)) {
          console.log(
            "[Tickets] Loaded",
            response.length,
            "tickets (direct array response)",
          );
          const transformedTickets = transformBackendTickets(response as any[]);
          setTickets(transformedTickets);
        }
        // Handle case where data is directly in response object as array
        else if (
          response &&
          typeof response === "object" &&
          Array.isArray((response as any).data)
        ) {
          console.log(
            "[Tickets] Loaded",
            (response as any).data.length,
            "tickets (response.data)",
          );
          const transformedTickets = transformBackendTickets(
            (response as any).data,
          );
          setTickets(transformedTickets);
        } else {
          console.warn("[Tickets] Unexpected response format:", response);
          setError("Unable to load tickets");
        }
      } catch (err) {
        const error =
          err instanceof Error ? err.message : "Failed to load tickets";
        console.error("[Tickets] Fetch error:", error);
        setError(error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchTickets();
  }, [isAuthenticated]);

  if (!isAuthenticated) {
    return null;
  }

  // Filter and search tickets
  const filteredTickets = tickets
    .filter((ticket) => {
      // Ensure status is properly normalized for comparison
      const ticketStatus = ticket.status?.toLowerCase() || "pending";
      const filterStatus = statusFilter?.toLowerCase() || "all";

      const matchesStatus =
        filterStatus === "all" || ticketStatus === filterStatus;
      const matchesPriority =
        priorityFilter === "all" || ticket.priority === priorityFilter;
      const matchesSearch =
        searchTerm === "" ||
        ticket.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        ticket.description.toLowerCase().includes(searchTerm.toLowerCase());

      return matchesStatus && matchesPriority && matchesSearch;
    })
    .sort((a, b) => {
      // Define priority order: high (0) > medium (1) > low (2)
      const priorityOrder: { [key: string]: number } = {
        high: 0,
        medium: 1,
        low: 2,
      };

      const priorityA = priorityOrder[a.priority?.toLowerCase() || "low"] ?? 2;
      const priorityB = priorityOrder[b.priority?.toLowerCase() || "low"] ?? 2;

      // If priorities are equal, sort by created date (newest first)
      if (priorityA === priorityB) {
        return (
          new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
        );
      }

      return priorityA - priorityB;
    });

  const pendingCount = tickets.filter(
    (t) => t.status?.toLowerCase() === "pending",
  ).length;
  const resolvedCount = tickets.filter(
    (t) => t.status?.toLowerCase() === "resolved",
  ).length;

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "high":
        return "bg-red-500 bg-opacity-10 border-red-500 text-red-400";
      case "medium":
        return "bg-yellow-500 bg-opacity-10 border-yellow-500 text-yellow-400";
      case "low":
        return "bg-green-500 bg-opacity-10 border-green-500 text-green-400";
      default:
        return "bg-slate-500 bg-opacity-10 border-slate-500 text-slate-400";
    }
  };

  const getStatusIcon = (status: string) => {
    if (status === "resolved") {
      return (
        <div className="w-6 h-6 bg-green-500 bg-opacity-20 rounded-full flex items-center justify-center">
          <svg
            className="w-4 h-4 text-green-400"
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path
              fillRule="evenodd"
              d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
              clipRule="evenodd"
            />
          </svg>
        </div>
      );
    }
    return (
      <div className="w-6 h-6 bg-yellow-500 bg-opacity-20 rounded-full flex items-center justify-center">
        <svg
          className="w-4 h-4 text-yellow-400"
          fill="currentColor"
          viewBox="0 0 20 20"
        >
          <path d="M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 100-2 1 1 0 000 2zm6 0a1 1 0 100-2 1 1 0 000 2z" />
        </svg>
      </div>
    );
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-8 gap-4">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2">Tickets</h1>
            <p className="text-slate-400">
              Manage and track all your support tickets
            </p>
          </div>
          <Link href="/tickets/create">
            <Button className="bg-blue-600 hover:bg-blue-700 text-white font-medium">
              Create Ticket
            </Button>
          </Link>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <Card className="bg-slate-800 border-slate-700 p-4">
            <p className="text-slate-400 text-sm">Total Tickets</p>
            <p className="text-2xl font-bold text-white">{tickets.length}</p>
          </Card>
          <Card className="bg-slate-800 border-slate-700 p-4">
            <p className="text-slate-400 text-sm">Pending</p>
            <p className="text-2xl font-bold text-yellow-400">{pendingCount}</p>
          </Card>
          <Card className="bg-slate-800 border-slate-700 p-4">
            <p className="text-slate-400 text-sm">Resolved</p>
            <p className="text-2xl font-bold text-green-400">{resolvedCount}</p>
          </Card>
        </div>

        {/* Filters */}
        <Card className="bg-slate-800 border-slate-700 p-6 mb-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-200 mb-2">
                Search
              </label>
              <input
                type="text"
                placeholder="Search tickets..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full bg-slate-700 border border-slate-600 text-white placeholder-slate-400 rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-600"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-200 mb-2">
                Status
              </label>
              <select
                value={statusFilter}
                onChange={(e) =>
                  setStatusFilter(e.target.value as StatusFilter)
                }
                className="w-full bg-slate-700 border border-slate-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-600"
              >
                <option value="all">All Status</option>
                <option value="pending">Pending</option>
                <option value="resolved">Resolved</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-200 mb-2">
                Priority
              </label>
              <select
                value={priorityFilter}
                onChange={(e) =>
                  setPriorityFilter(e.target.value as PriorityFilter)
                }
                className="w-full bg-slate-700 border border-slate-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-600"
              >
                <option value="all">All Priorities</option>
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>
            <div className="flex items-end">
              <Button
                onClick={() => {
                  setStatusFilter("all");
                  setPriorityFilter("all");
                  setSearchTerm("");
                }}
                className="w-full bg-slate-700 hover:bg-slate-600 text-white font-medium"
              >
                Reset Filters
              </Button>
            </div>
          </div>
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

        {/* Tickets List */}
        {!isLoading && (
          <div className="space-y-4">
            {filteredTickets.length > 0 ? (
              filteredTickets.map((ticket) => (
                <Link key={ticket.id} href={`/tickets/${ticket.id}`}>
                  <Card className="bg-slate-800 border-slate-700 p-6 hover:border-slate-600 transition-colors cursor-pointer">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          {getStatusIcon(ticket.status)}
                          <h3 className="text-lg font-semibold text-white">
                            {ticket.title}
                          </h3>
                        </div>
                        <p className="text-slate-400 text-sm mb-3 line-clamp-2">
                          {ticket.description}
                        </p>
                        <div className="flex flex-wrap gap-2 items-center">
                          <span className="text-xs bg-blue-500 bg-opacity-20 text-blue-300 px-3 py-1 rounded-full">
                            {ticket.category}
                          </span>
                          <span
                            className={`text-xs px-3 py-1 rounded-full border ${getPriorityColor(ticket.priority)}`}
                          >
                            {ticket.priority.charAt(0).toUpperCase() +
                              ticket.priority.slice(1)}{" "}
                            Priority
                          </span>
                          {ticket.categoryConfidence > 0 && (
                            <span className="text-xs text-slate-400">
                              Confidence:{" "}
                              {(ticket.categoryConfidence * 100).toFixed(1)}%
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="text-right text-sm text-slate-400">
                        <p>{new Date(ticket.createdAt).toLocaleDateString()}</p>
                      </div>
                    </div>
                  </Card>
                </Link>
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
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
                <p className="text-slate-400 text-lg">No tickets found</p>
                <p className="text-slate-500 text-sm mt-1">
                  Try adjusting your filters or create a new ticket
                </p>
              </Card>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
