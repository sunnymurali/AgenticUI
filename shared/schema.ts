import { pgTable, text, serial, real, json, timestamp } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const agents = pgTable("agents", {
  id: serial("id").primaryKey(),
  agent_id: text("agent_id").notNull().unique(),
  name: text("name").notNull(),
  model_name: text("model_name").notNull(),
  system_prompt: text("system_prompt").notNull(),
  temperature: real("temperature").notNull(),
  retriever_strategy: text("retriever_strategy"),
  interactions: json("interactions").$type<AgentInteraction[]>().default([]),
  created_at: timestamp("created_at").defaultNow(),
  updated_at: timestamp("updated_at").defaultNow(),
});

export const sessions = pgTable("sessions", {
  id: serial("id").primaryKey(),
  session_id: text("session_id").notNull(),
  agent_id: text("agent_id").notNull(),
  messages: json("messages").$type<ChatMessage[]>().default([]),
  updated_at: timestamp("updated_at").defaultNow(),
});

export const documents = pgTable("documents", {
  id: serial("id").primaryKey(),
  doc_id: text("doc_id").notNull(),
  agent_id: text("agent_id").notNull(),
  file_name: text("file_name").notNull(),
  content: text("content").notNull(),
  upload_date: timestamp("upload_date").defaultNow(),
});

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp?: string;
}

export interface AgentInteraction {
  session_id: string;
  user: string;
  assistant: string;
  timestamp: string;
}

export const insertAgentSchema = createInsertSchema(agents).omit({
  id: true,
  agent_id: true,
  interactions: true,
  created_at: true,
  updated_at: true,
}).extend({
  temperature: z.number().min(0).max(1),
  retriever_strategy: z.string().optional(),
});

export const chatRequestSchema = z.object({
  session_id: z.string().optional(),
  message: z.string().min(1),
  doc_id: z.string().optional(),
  file_name: z.string().optional(),
});

export const searchDocsSchema = z.object({
  query: z.string().min(1),
  top_k: z.number().min(1).max(20).default(3),
});

export type InsertAgent = z.infer<typeof insertAgentSchema>;
export type Agent = typeof agents.$inferSelect;
export type Session = typeof sessions.$inferSelect;
export type Document = typeof documents.$inferSelect;
export type ChatRequest = z.infer<typeof chatRequestSchema>;
export type SearchDocsRequest = z.infer<typeof searchDocsSchema>;
