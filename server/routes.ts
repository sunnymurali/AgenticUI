import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { insertAgentSchema, chatRequestSchema, searchDocsSchema } from "@shared/schema";
import { z } from "zod";

export async function registerRoutes(app: Express): Promise<Server> {
  // Agent routes
  app.post("/api/agents", async (req, res) => {
    try {
      const validatedData = insertAgentSchema.parse(req.body);
      const agent = await storage.createAgent(validatedData);
      res.json(agent);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: "Validation error", details: error.errors });
      } else {
        res.status(500).json({ error: "Failed to create agent" });
      }
    }
  });

  app.get("/api/agents", async (req, res) => {
    try {
      const agents = await storage.getAllAgents();
      // Add document counts for each agent
      const agentsWithDocs = await Promise.all(
        agents.map(async (agent) => {
          const documents = await storage.getDocumentsByAgent(agent.agent_id);
          return {
            ...agent,
            document_count: documents.length,
            documents: [...new Set(documents.map(doc => doc.file_name))], // unique filenames
          };
        })
      );
      res.json(agentsWithDocs);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch agents" });
    }
  });

  app.get("/api/agents/:agentId", async (req, res) => {
    try {
      const agent = await storage.getAgent(req.params.agentId);
      if (!agent) {
        return res.status(404).json({ error: "Agent not found" });
      }
      
      const documents = await storage.getDocumentsByAgent(agent.agent_id);
      res.json({
        agent,
        documents: [...new Set(documents.map(doc => doc.file_name))],
        document_count: documents.length,
      });
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch agent" });
    }
  });

  app.patch("/api/agents/:agentId", async (req, res) => {
    try {
      const updates = insertAgentSchema.partial().parse(req.body);
      const agent = await storage.updateAgent(req.params.agentId, updates);
      if (!agent) {
        return res.status(404).json({ error: "Agent not found" });
      }
      res.json(agent);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: "Validation error", details: error.errors });
      } else {
        res.status(500).json({ error: "Failed to update agent" });
      }
    }
  });

  app.delete("/api/agents/:agentId", async (req, res) => {
    try {
      const deleted = await storage.deleteAgent(req.params.agentId);
      if (!deleted) {
        return res.status(404).json({ error: "Agent not found" });
      }
      res.json({ message: "Agent deleted successfully" });
    } catch (error) {
      res.status(500).json({ error: "Failed to delete agent" });
    }
  });

  // Chat route - proxy to FastAPI backend
  app.post("/api/chat/:agentId", async (req, res) => {
    try {
      const validatedData = chatRequestSchema.parse(req.body);
      
      // In a real implementation, this would proxy to the FastAPI backend
      // For now, we'll simulate a response
      const agent = await storage.getAgent(req.params.agentId);
      if (!agent) {
        return res.status(404).json({ error: "Agent not found" });
      }

      const sessionId = validatedData.session_id || `session_${Date.now()}`;
      
      // Store/update session
      const existingSession = await storage.getSession(sessionId, req.params.agentId);
      const messages = existingSession?.messages || [];
      messages.push({ role: "user", content: validatedData.message, timestamp: new Date().toISOString() });
      
      // Simulate AI response
      const aiResponse = `I understand you're asking about "${validatedData.message}". As ${agent.name}, I'm here to help based on my configuration with ${agent.model_name} at temperature ${agent.temperature}.`;
      messages.push({ role: "assistant", content: aiResponse, timestamp: new Date().toISOString() });
      
      if (existingSession) {
        await storage.updateSession(sessionId, req.params.agentId, messages);
      } else {
        await storage.createSession({
          session_id: sessionId,
          agent_id: req.params.agentId,
          messages,
        });
      }

      res.json({
        session_id: sessionId,
        response: aiResponse,
      });
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: "Validation error", details: error.errors });
      } else {
        res.status(500).json({ error: "Failed to process chat message" });
      }
    }
  });

  // Document upload simulation
  app.post("/api/agents/:agentId/upload-docs", async (req, res) => {
    try {
      // In real implementation, this would handle file upload and process PDFs
      res.json({ 
        message: "Document upload would be handled by FastAPI backend",
        uploaded_chunks: 10 
      });
    } catch (error) {
      res.status(500).json({ error: "Failed to upload document" });
    }
  });

  // Stats endpoint
  app.get("/api/stats", async (req, res) => {
    try {
      const stats = await storage.getDocumentStats();
      res.json(stats);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch stats" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
