import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export function useAgentDocuments(agentId: string) {
  return useQuery({
    queryKey: ['/api/agent', agentId, 'documents'],
    queryFn: () => api.getAgentDocuments(agentId),
    enabled: !!agentId, // Only run query if agentId is provided
  });
}