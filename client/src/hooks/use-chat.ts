import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { ChatRequest } from "@shared/schema";
import type { ChatMessage } from "@/lib/types";
import { useToast } from "@/hooks/use-toast";

export function useChat(agentId: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const { toast } = useToast();

  const sendMessageMutation = useMutation({
    mutationFn: (data: ChatRequest) => api.sendMessage(agentId, data),
    onSuccess: (response) => {
      setSessionId(response.session_id);
      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          content: response.response,
          timestamp: new Date().toISOString(),
        }
      ]);
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message || "Failed to send message",
        variant: "destructive",
      });
    },
  });

  const sendMessage = (content: string, fileName?: string) => {
    // Add user message immediately
    const userMessage: ChatMessage = {
      role: "user",
      content,
      timestamp: new Date().toISOString(),
    };
    setMessages(prev => [...prev, userMessage]);
    // Send to API with optional file_name
    sendMessageMutation.mutate({
      session_id: sessionId,
      message: content,
      file_name: fileName,
    });
  };

  const clearChat = () => {
    setMessages([]);
    setSessionId(undefined);
  };

  return {
    messages,
    sessionId,
    sendMessage,
    clearChat,
    isLoading: sendMessageMutation.isPending,
  };
}
