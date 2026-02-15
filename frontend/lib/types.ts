export interface User {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  googleId?: string;
  createdAt: string;
  updatedAt: string;
}

export interface HistoricalTicket {
  ticket_id: string;
  title: string;
  description: string;
  status: "Resolved" | "Open" | "Pending";
  resolution: string;
}

export interface KnowledgeBaseArticle {
  article_id: string;
  title: string;
  solution: string;
  category: string;
}

export interface MLClassification {
  error: string;
  message: string;
  recommendations: string[];
  timestamp: string;
}

export interface MLCategoryPrediction {
  predicted: string;
  confidence: number;
  top_3_predictions: Array<{ category: string; confidence: number }>;
}

export interface MLDuplicatePrediction {
  has_duplicates: boolean;
  count: number;
  similar_tickets: any[];
}

export interface MLPriorityPrediction {
  predicted: string;
  impact: string;
  urgency: string;
  confidence: number;
}

export interface MLResolverPrediction {
  assigned_to: string;
  confidence: number;
}

export interface MLPredictions {
  category: MLCategoryPrediction;
  duplicates: MLDuplicatePrediction;
  priority: MLPriorityPrediction;
  resolver_group: MLResolverPrediction;
}

export interface MLAutoResponse {
  draft: string;
  confidence: number;
  kb_incorporated: boolean;
}

export interface MLKnowledgeBase {
  articles_found: number;
  has_solution: boolean;
  articles: any[];
}

export interface MLPatterns {
  detected: boolean;
  type: string | null;
  insights: string;
}

export interface MLProactiveInsights {
  insights: string[];
  count: number;
  has_critical: boolean;
}

export interface MLRagInsights {
  auto_response: MLAutoResponse;
  knowledge_base: MLKnowledgeBase;
  patterns: MLPatterns;
  proactive_insights: MLProactiveInsights;
}

export interface MLAuditTrail {
  category_reasoning: any;
  duplicate_reasoning: any;
  kb_reasoning: any;
  priority_reasoning: any;
  resolver_reasoning: any;
}

export interface MLPredictResponse {
  ticket_id: string;
  predictions: MLPredictions;
  rag_insights: MLRagInsights;
  audit_trail: MLAuditTrail;
  processing_time_ms: number;
  timestamp: string;
}

export interface Ticket {
  id: string;
  title: string;
  description: string;
  status: "pending" | "resolved";
  closed?: boolean; // True if status is 'pending' or 'resolved'
  priority: "low" | "medium" | "high";
  category: string;
  categoryConfidence: number;
  assignee?: string;
  createdBy?: string;
  createdAt: string;
  updatedAt?: string;
  mlPrediction?: MLPredictionResult;
  user?: string;
  email?: string;
  ticket_id?: string;
  historical_tickets?: HistoricalTicket[];
  knowledge_base?: KnowledgeBaseArticle[];
  ml_classification?: MLClassification;
  resolver_group?: string;
}

export interface MLPredictionResult {
  predicted_class: string;
  confidence_scores: {
    [key: string]: number;
  };
  processed_text: string;
  timestamp: string;
}

export interface MLPredictionInput {
  text: string;
}

export interface Notification {
  id: string;
  userId: string;
  title: string;
  message: string;
  type: "ticket_created" | "ticket_updated" | "ticket_resolved";
  ticketId?: string;
  read: boolean;
  createdAt: string;
}

export interface ApiResponse<T> {
  success: boolean;
  message: string;
  data?: T;
  error?: string;
}

export interface AuthResponse {
  token: string;
  user: User;
}

export interface LoginPayload {
  email: string;
  password: string;
}

export interface SignupPayload {
  email: string;
  password: string;
  name: string;
}

export interface GoogleLoginPayload {
  idToken: string;
}

export interface CreateTicketPayload {
  title: string;
  description: string;
  priority: "low" | "medium" | "high";
}
