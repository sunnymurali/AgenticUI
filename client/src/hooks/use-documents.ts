import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

export function useAgentDocuments(agentId: string) {
  return useQuery({
    queryKey: ['/api/agent', agentId, 'documents'],
    queryFn: () => api.getAgentDocuments(agentId),
    enabled: !!agentId, // Only run query if agentId is provided
  });
}

export function useDeleteAgentDocument(agentId: string) {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  return useMutation({
    mutationFn: (fileName: string) => api.deleteAgentDocument(agentId, fileName),
    onSuccess: (_, fileName) => {
      // Invalidate documents cache to refresh the list
      queryClient.invalidateQueries({ queryKey: ['/api/agent', agentId, 'documents'] });
      queryClient.invalidateQueries({ queryKey: ['/api/agents'] });
      toast({
        title: "Success",
        description: `Document "${fileName}" deleted successfully`,
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Error", 
        description: error.message || "Failed to delete document",
        variant: "destructive",
      });
    },
  });
}

export function useSearchAgentDocuments(agentId: string) {
  const { toast } = useToast();

  return useMutation({
    mutationFn: ({ query, topK }: { query: string; topK?: number }) => 
      api.searchAgentDocuments(agentId, query, topK),
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message || "Failed to search documents", 
        variant: "destructive",
      });
    },
  });
}