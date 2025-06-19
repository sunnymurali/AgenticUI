import { agents, sessions, documents, type Agent, type InsertAgent, type Session, type Document } from "@shared/schema";

export interface IStorage {
  // Agent operations
  createAgent(agent: InsertAgent): Promise<Agent>;
  getAgent(agentId: string): Promise<Agent | undefined>;
  getAllAgents(): Promise<Agent[]>;
  updateAgent(agentId: string, updates: Partial<InsertAgent>): Promise<Agent | undefined>;
  deleteAgent(agentId: string): Promise<boolean>;
  
  // Session operations
  createSession(session: { session_id: string; agent_id: string; messages: any[] }): Promise<Session>;
  getSession(sessionId: string, agentId: string): Promise<Session | undefined>;
  updateSession(sessionId: string, agentId: string, messages: any[]): Promise<Session | undefined>;
  
  // Document operations
  getDocumentsByAgent(agentId: string): Promise<Document[]>;
  getDocumentStats(): Promise<{ totalDocuments: number; totalAgents: number; totalSessions: number; totalQueries: number }>;
}

export class MemStorage implements IStorage {
  private agents: Map<string, Agent>;
  private sessions: Map<string, Session>;
  private documents: Map<string, Document>;
  private currentId: number;

  constructor() {
    this.agents = new Map();
    this.sessions = new Map();
    this.documents = new Map();
    this.currentId = 1;
  }

  async createAgent(insertAgent: InsertAgent): Promise<Agent> {
    const id = this.currentId++;
    const agent_id = `agent_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const agent: Agent = {
      ...insertAgent,
      id,
      agent_id,
      interactions: [],
      created_at: new Date(),
      updated_at: new Date(),
    };
    this.agents.set(agent_id, agent);
    return agent;
  }

  async getAgent(agentId: string): Promise<Agent | undefined> {
    return this.agents.get(agentId);
  }

  async getAllAgents(): Promise<Agent[]> {
    return Array.from(this.agents.values());
  }

  async updateAgent(agentId: string, updates: Partial<InsertAgent>): Promise<Agent | undefined> {
    const agent = this.agents.get(agentId);
    if (!agent) return undefined;
    
    const updatedAgent: Agent = {
      ...agent,
      ...updates,
      updated_at: new Date(),
    };
    this.agents.set(agentId, updatedAgent);
    return updatedAgent;
  }

  async deleteAgent(agentId: string): Promise<boolean> {
    return this.agents.delete(agentId);
  }

  async createSession(sessionData: { session_id: string; agent_id: string; messages: any[] }): Promise<Session> {
    const id = this.currentId++;
    const session: Session = {
      id,
      ...sessionData,
      updated_at: new Date(),
    };
    this.sessions.set(`${sessionData.session_id}_${sessionData.agent_id}`, session);
    return session;
  }

  async getSession(sessionId: string, agentId: string): Promise<Session | undefined> {
    return this.sessions.get(`${sessionId}_${agentId}`);
  }

  async updateSession(sessionId: string, agentId: string, messages: any[]): Promise<Session | undefined> {
    const key = `${sessionId}_${agentId}`;
    const session = this.sessions.get(key);
    if (!session) return undefined;
    
    const updatedSession: Session = {
      ...session,
      messages,
      updated_at: new Date(),
    };
    this.sessions.set(key, updatedSession);
    return updatedSession;
  }

  async getDocumentsByAgent(agentId: string): Promise<Document[]> {
    return Array.from(this.documents.values()).filter(doc => doc.agent_id === agentId);
  }

  async getDocumentStats(): Promise<{ totalDocuments: number; totalAgents: number; totalSessions: number; totalQueries: number }> {
    return {
      totalDocuments: this.documents.size,
      totalAgents: this.agents.size,
      totalSessions: this.sessions.size,
      totalQueries: Array.from(this.agents.values()).reduce((sum, agent) => sum + (agent.interactions?.length || 0), 0),
    };
  }
}

export const storage = new MemStorage();
