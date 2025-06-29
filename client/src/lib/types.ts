// ADDED: Tool-related fields to AgentWithStats
export interface AgentWithStats {
  id: number;
  agent_id: string;
  name: string;
  model_name: string;
  system_prompt: string;
  temperature: number;
  retriever_strategy?: string;
  interactions?: AgentInteraction[];
  created_at: Date;
  updated_at: Date;
  document_count: number;
  documents: string[];
  use_tools?: boolean;
  tools?: any[]; // Array of tools from backend  // NEW
  // Tool-specific fields (present when tool_id is not null)
  tool_id?: string;
  tool_name?: string;
  tool_description?: string;
  endpoint_url?: string;
  api_token?: string;
  tool_parameters?: string;
}

// ADDED: New ToolDefinition interface
export interface ToolDefinition {
  name: string;
  description: string;
  endpoint_url: string;
  api_token: string;
  parameters?: Record<string, any>;
}

export interface AgentInteraction {
  session_id: string;
  user: string;
  assistant: string;
  timestamp: string;
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp?: string;
}

export interface ChatSession {
  id: number;
  session_id: string;
  agent_id: string;
  messages: ChatMessage[];
  updated_at: Date;
}

export interface DocumentInfo {
  id: number;
  doc_id: string;
  agent_id: string;
  file_name: string;
  content: string;
  upload_date: Date;
}

export interface Stats {
  totalAgents: number;
  totalSessions: number;
  totalDocuments: number;
  totalQueries: number;
}

export interface ChatResponse {
  session_id: string;
  response: string;
}

export interface UploadResponse {
  message: string;
  uploaded_chunks: number;
}
