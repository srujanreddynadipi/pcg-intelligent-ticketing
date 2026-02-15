"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import { useTokenRestoration } from "@/hooks/use-token-restoration";
import { apiClient, mlApiClient } from "@/lib/api";
import { API_ENDPOINTS } from "@/lib/config";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import {
  MLPredictResponse,
  HistoricalTicket,
  KnowledgeBaseArticle,
  Ticket,
} from "@/lib/types";

export default function CreateTicketPage() {
  const router = useRouter();
  const { user, isAuthenticated } = useAuth();
  useTokenRestoration();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const [mlResult, setMlResult] = useState<MLPredictResponse | null>(null);
  const [formData, setFormData] = useState({
    title: "",
    description: "",
    priority: "medium" as "low" | "medium" | "high",
  });

  if (!isAuthenticated) {
    return null;
  }

  const handleChange = (
    e: React.ChangeEvent<
      HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement
    >,
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleGetMLPrediction = async () => {
    if (!formData.description.trim()) {
      setError("Please enter a description first");
      return;
    }

    setIsLoading(true);
    setError("");
    try {
      console.log("[Ticket] Starting ML prediction with new schema...");

      // Prepare ML prediction payload with the new format
      const mlPayload = {
        user: user?.email || "",
        title: formData.title || "Untitled",
        description: formData.description,
        historical_tickets: [], // Can be populated from backend if needed
        knowledge_base: [], // Can be populated from backend if needed
      };

      console.log("[Ticket] Calling ML API with payload:", mlPayload);

      const result = await mlApiClient.predict<MLPredictResponse>(mlPayload);

      console.log("[Ticket] ML API response:", result);

      // Handle the actual ML response format
      if (result.ticket_id && result.predictions) {
        setMlResult(result);
      } else {
        setError("Unexpected ML response format");
      }
    } catch (err) {
      setError(
        "Failed to get AI prediction. Please check ML service is running.",
      );
      console.error("[Ticket] ML prediction error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError("");

    try {
      // Build the complete ticket payload with new schema
      const payload = {
        user: user?.email || "",
        title: formData.title,
        description: formData.description,
        priority:
          mlResult?.predictions?.priority?.predicted || formData.priority,
        category: mlResult?.predictions?.category?.predicted || "Uncategorized",
        resolver_group:
          mlResult?.predictions?.resolver_group?.assigned_to || "",
        email: user?.email || "",
        historical_tickets: [],
        knowledge_base: [],
        ml_classification: mlResult || undefined,
      };

      console.log("[Ticket] Creating ticket with payload:", payload);

      const response = await apiClient.post<Ticket>(
        API_ENDPOINTS.TICKETS.CREATE,
        payload,
      );

      console.log("[Ticket] Create response:", response);

      // Check success indicators - response.success should be true for successful creation
      if (response.success) {
        console.log("[Ticket] Ticket created successfully!");
        // Show success toast/message (if you have toast notifications)
        alert(
          `✅ Ticket created successfully!\nTicket ID: ${response.data?.ticket_id || response.data?.id || "CREATED"}`,
        );
        router.push("/");
      } else {
        setError(
          response.error || response.message || "Failed to create ticket",
        );
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create ticket");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            Create New Ticket
          </h1>
          <p className="text-slate-400">
            Describe your issue and our AI will automatically categorize it
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Title */}
          <Card className="bg-slate-800 border-slate-700 p-6">
            <label
              htmlFor="title"
              className="block text-sm font-medium text-slate-200 mb-3"
            >
              Ticket Title
            </label>
            <Input
              id="title"
              name="title"
              type="text"
              placeholder="Brief summary of the issue"
              value={formData.title}
              onChange={handleChange}
              required
              disabled={isLoading}
              className="bg-slate-700 border-slate-600 text-white placeholder-slate-400"
            />
          </Card>

          {/* Description */}
          <Card className="bg-slate-800 border-slate-700 p-6">
            <label
              htmlFor="description"
              className="block text-sm font-medium text-slate-200 mb-3"
            >
              Description
            </label>
            <textarea
              id="description"
              name="description"
              placeholder="Provide detailed information about the issue..."
              value={formData.description}
              onChange={handleChange}
              required
              disabled={isLoading}
              rows={8}
              className="w-full bg-slate-700 border border-slate-600 text-white placeholder-slate-400 rounded-md px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-600"
            />
            <p className="text-slate-400 text-sm mt-2">
              The more details you provide, the better our AI can categorize
              your issue
            </p>
          </Card>

          {/* Priority */}
          <Card className="bg-slate-800 border-slate-700 p-6">
            <label
              htmlFor="priority"
              className="block text-sm font-medium text-slate-200 mb-3"
            >
              Priority Level
            </label>
            <select
              id="priority"
              name="priority"
              value={formData.priority}
              onChange={handleChange}
              disabled={isLoading}
              className="w-full bg-slate-700 border border-slate-600 text-white rounded-md px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-600"
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </Card>

          {/* ML Prediction */}
          <Card className="bg-slate-800 border-slate-700 p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold text-white">
                  AI Categorization
                </h3>
                <p className="text-slate-400 text-sm">
                  Let our machine learning model categorize your ticket
                </p>
              </div>
              <Button
                type="button"
                onClick={handleGetMLPrediction}
                disabled={isLoading || !formData.description.trim()}
                className="bg-purple-600 hover:bg-purple-700 text-white font-medium"
              >
                {isLoading ? "Analyzing..." : "Get AI Prediction"}
              </Button>
            </div>

            {mlResult && (
              <div className="space-y-4 mt-4">
                <div className="bg-slate-700 rounded-lg p-4 space-y-4">
                  {/* Category Prediction */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-slate-300">
                        Predicted Category:
                      </span>
                      <span className="text-lg font-bold text-blue-400">
                        {mlResult.predictions.category.predicted}
                      </span>
                    </div>
                    <div className="w-full bg-slate-600 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{
                          width: `${(mlResult.predictions.category.confidence || 0) * 100}%`,
                        }}
                      />
                    </div>
                    <p className="text-slate-400 text-sm mt-1">
                      {(
                        (mlResult.predictions.category.confidence || 0) * 100
                      ).toFixed(1)}
                      % confidence
                    </p>
                  </div>

                  {/* Priority Prediction */}
                  <div className="border-t border-slate-600 pt-3">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-300">
                        Predicted Priority:
                      </span>
                      <span className="text-lg font-bold text-amber-400">
                        {mlResult.predictions.priority.predicted}
                      </span>
                    </div>
                    <p className="text-slate-400 text-sm mt-1">
                      Impact: {mlResult.predictions.priority.impact} | Urgency:{" "}
                      {mlResult.predictions.priority.urgency}
                    </p>
                  </div>

                  {/* Resolver Group */}
                  <div className="border-t border-slate-600 pt-3">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-300">Assigned Group:</span>
                      <span className="text-lg font-bold text-green-400">
                        {mlResult.predictions.resolver_group.assigned_to}
                      </span>
                    </div>
                    <p className="text-slate-400 text-sm mt-1">
                      {(
                        (mlResult.predictions.resolver_group.confidence || 0) *
                        100
                      ).toFixed(1)}
                      % confidence
                    </p>
                  </div>

                  {/* Auto Response / Recommendations */}
                  {mlResult.rag_insights?.auto_response?.draft && (
                    <div className="border-t border-slate-600 pt-3">
                      <span className="text-slate-300 block mb-2">
                        Suggested Resolution Draft:
                      </span>
                      <p className="text-slate-300 text-sm whitespace-pre-wrap">
                        {mlResult.rag_insights.auto_response.draft}
                      </p>
                    </div>
                  )}

                  {/* Proactive Insights */}
                  {mlResult.rag_insights?.proactive_insights?.insights &&
                    mlResult.rag_insights.proactive_insights.insights.length >
                      0 && (
                      <div className="border-t border-slate-600 pt-3">
                        <span className="text-slate-300 block mb-2">
                          Insights:
                        </span>
                        <ul className="space-y-2">
                          {mlResult.rag_insights.proactive_insights.insights.map(
                            (insight, idx) => {
                              // Handle both string and object formats
                              const insightText =
                                typeof insight === "string"
                                  ? insight
                                  : (insight as any)?.message ||
                                    (insight as any)?.recommendation ||
                                    JSON.stringify(insight);

                              return (
                                <li
                                  key={idx}
                                  className="text-sm text-slate-400 flex items-start gap-2 bg-slate-600 bg-opacity-50 p-2 rounded"
                                >
                                  <span className="text-blue-400 mt-0.5 flex-shrink-0">
                                    •
                                  </span>
                                  <div className="flex-1">
                                    <span>{insightText}</span>
                                  </div>
                                </li>
                              );
                            },
                          )}
                        </ul>
                      </div>
                    )}
                </div>

                {/* Processing Info */}
                <div className="text-xs text-slate-500 text-right">
                  Processed in {mlResult.processing_time_ms?.toFixed(2)}ms •
                  Ticket ID: {mlResult.ticket_id}
                </div>
              </div>
            )}
          </Card>

          {error && (
            <Card className="bg-red-500 bg-opacity-10 border border-red-500 p-4">
              <p className="text-red-400">{error}</p>
            </Card>
          )}

          {/* Action Buttons */}
          <div className="flex gap-4">
            <Button
              type="submit"
              disabled={isLoading}
              className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-medium py-3"
            >
              {isLoading ? "Creating..." : "Create Ticket"}
            </Button>
            <Button
              type="button"
              onClick={() => router.push("/tickets")}
              disabled={isLoading}
              className="flex-1 bg-slate-700 hover:bg-slate-600 text-white font-medium py-3"
            >
              Cancel
            </Button>
          </div>
        </form>
      </div>
    </main>
  );
}
