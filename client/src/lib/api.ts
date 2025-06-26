import { apiRequest } from "./queryClient";
import type { InsertAgent, ChatRequest, SearchDocsRequest } from "@shared/schema";
import type { AgentWithStats, Stats, ChatResponse, UploadResponse } from "./types";

const PYTHON_API_BASE = "http://localhost:8000";

// Helper function to check if Python backend is available
async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${PYTHON_API_BASE}/agents`, { 
      signal: AbortSignal.timeout(2000) 
    });
    return response.ok;
  } catch {
    return false;
  }
}

export const api = {
  // Agent operations
  async createAgent(data: InsertAgent): Promise<AgentWithStats> {
    const response = await fetch(`${PYTHON_API_BASE}/agent`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    
    if (!response.ok) {
      throw new Error(`Failed to create agent: ${response.statusText}`);
    }
    
    const result = await response.json();
    // Transform Python response to match frontend interface
    return {
      id: 0,
      agent_id: result.agent_id,
      name: data.name,
      model_name: data.model_name,
      system_prompt: data.system_prompt,
      temperature: data.temperature,
      retriever_strategy: data.retriever_strategy,
      use_tools: data.use_tools || false,
      interactions: [],
      created_at: new Date(),
      updated_at: new Date(),
      document_count: 0,
      documents: []
    };
  },

  async getAgents(): Promise<AgentWithStats[]> {
    const isBackendAvailable = await checkBackendHealth();
    if (!isBackendAvailable) {
      console.warn("Python backend not available, returning empty agents list");
      return [];
    }
    
    const response = await fetch(`${PYTHON_API_BASE}/agents`);
    
    if (!response.ok) {
      throw new Error(`Failed to get agents: ${response.statusText}`);
    }
    
    const result = await response.json();
    
    // Transform the nested response format to match frontend expectations
    return result.map((item: any, index: number) => ({
      id: index + 1,
      agent_id: item.agent.agent_id,
      name: item.agent.name,
      model_name: item.agent.model_name,
      system_prompt: item.agent.system_prompt,
      temperature: item.agent.temperature,
      retriever_strategy: item.agent.retriever_strategy,
      use_tools: item.agent.tool_id ? true : false,
      interactions: [],
      created_at: new Date(),
      updated_at: new Date(),
      document_count: item.documents ? item.documents.length : 0,
      documents: item.documents || []
    }));
  },

  async getAgent(agentId: string): Promise<{ agent: AgentWithStats; documents: string[]; document_count: number }> {
    const response = await fetch(`${PYTHON_API_BASE}/agent/${agentId}`);
    
    if (!response.ok) {
      throw new Error(`Failed to get agent: ${response.statusText}`);
    }
    
    const result = await response.json();
    
    return {
      agent: {
        id: 0,
        agent_id: result.agent.agent_id,
        name: result.agent.name,
        model_name: result.agent.model_name,
        system_prompt: result.agent.system_prompt,
        temperature: result.agent.temperature,
        use_tools: result.agent.tool_id ? true : false,
        retriever_strategy: result.agent.retriever_strategy,
        interactions: result.agent.interactions || [],
        created_at: new Date(),
        updated_at: new Date(),
        document_count: result.documents.length,
        documents: result.documents,
        // Tool-specific fields (when tool_id is present)
        tool_id: result.agent.tool_id || undefined,
        tool_name: result.agent.tool_name || undefined,
        tool_description: result.agent.tool_description || undefined,
        endpoint_url: result.agent.endpoint_url || undefined,
        api_token: result.agent.api_token || undefined,
        tool_parameters: result.agent.tool_parameters || undefined,
      },
      documents: result.documents,
      document_count: result.documents.length
    };
  },

  async updateAgent(agentId: string, data: Partial<InsertAgent>): Promise<AgentWithStats> {
    // Prepare request body in the exact format expected by the backend
    const requestBody = {
      name: data.name,
      model_name: data.model_name,
      system_prompt: data.system_prompt,
      temperature: data.temperature,
      retriever_strategy: data.retriever_strategy || "",
      use_tools: data.use_tools || false
    };



    const response = await fetch(`${PYTHON_API_BASE}/agent/${agentId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });
    
    if (!response.ok) {
      throw new Error(`Failed to update agent: ${response.statusText}`);
    }
    
    const result = await response.json();
    
    // Transform response to match expected format
    return {
      id: 0,
      agent_id: result.agent_id || agentId,
      name: result.name || requestBody.name || "",
      model_name: result.model_name || requestBody.model_name || "",
      system_prompt: result.system_prompt || requestBody.system_prompt || "",
      temperature: result.temperature ?? requestBody.temperature ?? 0.7,
      retriever_strategy: result.retriever_strategy || requestBody.retriever_strategy || "",
      use_tools: result.use_tools ?? false,
      interactions: [],
      created_at: new Date(),
      updated_at: new Date(),
      document_count: result.document_count || 0,
      documents: result.documents || []
    };
  },

  async deleteAgent(agentId: string): Promise<{ message: string }> {
    const response = await fetch(`${PYTHON_API_BASE}/agent/${agentId}`, {
      method: "DELETE",
    });
    
    if (!response.ok) {
      throw new Error(`Failed to delete agent: ${response.statusText}`);
    }
    
    return response.json();
  },

  // Modified sendMessage to support conditional endpoints
  async sendMessage(agentId: string, data: ChatRequest): Promise<ChatResponse> {
    const isBackendAvailable = await checkBackendHealth();
    if (!isBackendAvailable) {
      throw new Error("Python backend is not available. Please check if the backend is running on port 8000.");
    }
    // Step 1: Get agent details to check if it has tools
    const agentResponse = await fetch(`${PYTHON_API_BASE}/agent/${agentId}`);
    if (!agentResponse.ok) {
      throw new Error(`Failed to get agent details: ${agentResponse.statusText}`);
    }
    
    const agentData = await agentResponse.json();
    const hasTools = agentData.agent.tool_id ? true : false;

    // Step 2: Route to appropriate chat endpoint based on tool presence
    const endpoint = hasTools ? `/chat-with-tool/${agentId}` : `/chat/${agentId}`;
    const response = await fetch(`${PYTHON_API_BASE}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    
    if (!response.ok) {
      const text = (await response.text()) || response.statusText;
      throw new Error(`${response.status}: ${text}`);
    }
    
    return response.json();
  },


  // NEW: Tool creation function
  // NEW: Tool creation function
  // NEW: Tool creation function
  async createTool(agentId: string, tool: any): Promise<any> {
    const isBackendAvailable = await checkBackendHealth();
    if (!isBackendAvailable) {
      throw new Error("Python backend is not available. Please check if the backend is running on port 8000.");
    }
    const response = await fetch(`${PYTHON_API_BASE}/agent/${agentId}/tools`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(tool),
    });
    
    if (!response.ok) {
      const text = (await response.text()) || response.statusText;
      throw new Error(`${response.status}: ${text}`);
    }
    
    return response.json();
  },

  // Document operations
  async uploadDocument(agentId: string, file: File): Promise<UploadResponse> {
    // Check if backend is available
    const isBackendAvailable = await checkBackendHealth();
    if (!isBackendAvailable) {
      throw new Error("Python backend is not available. Please ensure the backend is running on port 8000.");
    }

    const formData = new FormData();
    formData.append("file", file);
    
    const response = await fetch(`${PYTHON_API_BASE}/agent/${agentId}/upload-docs`, {
      method: "POST",
      body: formData,
    });
    
    if (!response.ok) {
      const text = (await response.text()) || response.statusText;
      throw new Error(`Upload failed (${response.status}): ${text}`);
    }
    
    return response.json();
  },

  // Stats
  async getStats(): Promise<Stats> {
    const response = await fetch(`${PYTHON_API_BASE}/stats`);
    
    if (!response.ok) {
      throw new Error(`Failed to get stats: ${response.statusText}`);
    }
    
    return response.json();
  },

  // Get documents for a specific agent
  async getAgentDocuments(agentId: string): Promise<Array<{file_name: string, count: number}>> {
    const response = await fetch(`${PYTHON_API_BASE}/agent/${agentId}/docs`);
    
    if (!response.ok) {
      throw new Error(`Failed to get agent documents: ${response.statusText}`);
    }
    
    return response.json();
  },

  async deleteAgentDocument(agentId: string, fileName: string): Promise<{ message: string }> {
    const response = await fetch(`${PYTHON_API_BASE}/agent/${agentId}/delete-document?file_name=${encodeURIComponent(fileName)}`, {
      method: "DELETE",
    });
    
    if (!response.ok) {
      throw new Error(`Failed to delete document: ${response.statusText}`);
    }
    
    return response.json();
  },

  // Search documents within an agent
  async searchAgentDocuments(agentId: string, query: string, topK: number = 3): Promise<any[]> {
    const response = await fetch(`${PYTHON_API_BASE}/agent/${agentId}/search-docs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK }),
    });
    
    if (!response.ok) {
      throw new Error(`Failed to search documents: ${response.statusText}`);
    }
    
    return response.json();
  },

  async getDocumentsByFileName(agentId: string): Promise<string[]> {
    const response = await fetch(`${PYTHON_API_BASE}/agent/${agentId}/docs`);
    
    if (!response.ok) {
      throw new Error(`Failed to get documents by filename: ${response.statusText}`);
    }
    
    const docs: Array<{ file_name: string; content_preview: string; upload_date?: string }> = await response.json();
    // Extract unique filenames from the response
    const fileNameSet = new Set<string>();
    docs.forEach(doc => fileNameSet.add(doc.file_name));
    const filenames = Array.from(fileNameSet);
    return filenames;
  },
};
