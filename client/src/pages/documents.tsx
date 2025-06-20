import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { UploadDropzone } from "@/components/upload-dropzone";
import { useAgents } from "@/hooks/use-agents";
import { useAgentDocuments, useDeleteAgentDocuments, useSearchAgentDocuments } from "@/hooks/use-documents";
import { Bot, FileText, Search, Trash2, Upload, Loader2 } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";

export default function Documents() {
  const [selectedAgentId, setSelectedAgentId] = useState<string>("");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchModalOpen, setSearchModalOpen] = useState(false);
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  
  const { data: agents, isLoading: agentsLoading } = useAgents();
  const { data: documents, isLoading: documentsLoading } = useAgentDocuments(selectedAgentId);
  const deleteDocumentsMutation = useDeleteAgentDocuments(selectedAgentId);
  const searchDocumentsMutation = useSearchAgentDocuments(selectedAgentId);

  const selectedAgent = agents?.find(agent => agent.agent_id === selectedAgentId);

  const handleAgentSelect = (agentId: string) => {
    setSelectedAgentId(agentId);
    setSearchResults([]); // Clear search results when switching agents
    setSearchQuery("");
  };

  const handleDeleteDocuments = () => {
    if (selectedAgentId && confirm("Are you sure you want to delete all documents for this agent? This action cannot be undone.")) {
      deleteDocumentsMutation.mutate();
    }
  };

  const handleSearchDocuments = async () => {
    if (!searchQuery.trim() || !selectedAgentId) return;
    
    setIsSearching(true);
    try {
      const results = await searchDocumentsMutation.mutateAsync({ 
        query: searchQuery, 
        topK: 5 
      });
      setSearchResults(results);
    } catch (error) {
      console.error("Search failed:", error);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  // Transform documents array to the format expected by the UI
  const documentList = documents ? documents.map((doc, index) => ({
    id: index + 1,
    file_name: doc.file_name,
    chunks_count: doc.count,
    upload_date: "Recently", // This would come from API if available
  })) : [];

  return (
    <div className="p-6 space-y-6">
      <div className="mb-6">
        <h2 className="text-2xl font-semibold text-slate-800 mb-2">Document Management</h2>
        <p className="text-slate-600">Upload and manage documents for your AI agents.</p>
      </div>

      {/* Agent Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Select Agent</CardTitle>
        </CardHeader>
        <CardContent>
          {agentsLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {[...Array(3)].map((_, i) => (
                <div key={i} className="p-4 border border-gray-200 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <Skeleton className="w-10 h-10 rounded-lg" />
                    <div>
                      <Skeleton className="h-4 w-24 mb-1" />
                      <Skeleton className="h-3 w-16" />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : agents && agents.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {agents.map((agent) => (
                <div
                  key={agent.agent_id}
                  className={`p-4 border rounded-lg cursor-pointer transition-all duration-200 ${
                    selectedAgentId === agent.agent_id
                      ? 'border-primary bg-blue-50'
                      : 'border-gray-200 hover:border-primary hover:bg-blue-50'
                  }`}
                  onClick={() => handleAgentSelect(agent.agent_id)}
                >
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 gradient-blue-teal rounded-lg flex items-center justify-center">
                      <Bot className="text-white" />
                    </div>
                    <div>
                      <h4 className="font-medium text-slate-800">{agent.name}</h4>
                      <p className="text-sm text-slate-500">
                        {agent.document_count} documents
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <Bot className="mx-auto h-12 w-12 text-slate-400 mb-4" />
              <p className="text-slate-600">No agents available. Create an agent first.</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Document Upload Section */}
      {selectedAgent && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Upload size={20} />
              <span>Upload Documents for {selectedAgent.name}</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <UploadDropzone agentId={selectedAgent.agent_id} />
          </CardContent>
        </Card>
      )}

      {/* Document List */}
      {selectedAgent && (
        <Card>
          <CardHeader className="border-b border-gray-200">
            <div className="flex justify-between items-center">
              <CardTitle>Uploaded Documents</CardTitle>
              {documentList.length > 0 && (
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={handleDeleteDocuments}
                  disabled={deleteDocumentsMutation.isPending}
                >
                  {deleteDocumentsMutation.isPending ? (
                    <>
                      <Loader2 size={16} className="mr-2 animate-spin" />
                      Deleting...
                    </>
                  ) : (
                    <>
                      <Trash2 size={16} className="mr-2" />
                      Delete All
                    </>
                  )}
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent className="p-6">
            {documentsLoading ? (
              <div className="space-y-4">
                {[...Array(3)].map((_, i) => (
                  <div key={i} className="flex items-center space-x-4 p-4 border border-gray-200 rounded-lg">
                    <Skeleton className="w-12 h-12 rounded-lg" />
                    <div className="flex-1">
                      <Skeleton className="h-4 w-48 mb-2" />
                      <Skeleton className="h-3 w-32" />
                    </div>
                  </div>
                ))}
              </div>
            ) : documentList.length > 0 ? (
              <div className="space-y-4">
                {documentList.map((doc) => (
                  <div
                    key={doc.id}
                    className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:shadow-sm transition-shadow duration-200"
                  >
                    <div className="flex items-center space-x-4">
                      <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center">
                        <FileText className="text-red-600" />
                      </div>
                      <div>
                        <h4 className="font-medium text-slate-800">{doc.file_name}</h4>
                        <p className="text-sm text-slate-500">
                          {doc.chunks_count} chunks • Uploaded {doc.upload_date}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Dialog open={searchModalOpen} onOpenChange={setSearchModalOpen}>
                        <DialogTrigger asChild>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="text-slate-400 hover:text-slate-600"
                          >
                            <Search size={16} />
                          </Button>
                        </DialogTrigger>
                        <DialogContent className="max-w-2xl">
                          <DialogHeader>
                            <DialogTitle>Search Documents</DialogTitle>
                          </DialogHeader>
                          <div className="space-y-4">
                            <div className="flex space-x-2">
                              <Input
                                placeholder="Search through documents..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && handleSearchDocuments()}
                              />
                              <Button 
                                onClick={handleSearchDocuments} 
                                disabled={!searchQuery.trim() || isSearching}
                              >
                                {isSearching ? (
                                  <>
                                    <Loader2 size={16} className="mr-2 animate-spin" />
                                    Searching...
                                  </>
                                ) : (
                                  <>
                                    <Search size={16} className="mr-2" />
                                    Search
                                  </>
                                )}
                              </Button>
                            </div>
                            <div className="max-h-96 overflow-y-auto space-y-4">
                              {searchResults.length > 0 ? (
                                searchResults.map((result, index) => (
                                  <div key={index} className="p-4 border border-gray-200 rounded-lg">
                                    <h4 className="font-medium text-slate-800 mb-2">
                                      {result.metadata?.source || result.file_name || "Unknown source"}
                                    </h4>
                                    <p className="text-sm text-slate-600 mb-2">
                                      {result.page_content || result.content || "No content available"}
                                    </p>
                                    <p className="text-xs text-slate-500">
                                      Score: {result.score?.toFixed(3) || "N/A"}
                                    </p>
                                  </div>
                                ))
                              ) : searchQuery && !isSearching ? (
                                <div className="text-center py-8">
                                  <FileText className="mx-auto h-12 w-12 text-slate-400 mb-4" />
                                  <p className="text-slate-500">No results found for "{searchQuery}"</p>
                                </div>
                              ) : !searchQuery ? (
                                <div className="text-center py-8">
                                  <Search className="mx-auto h-12 w-12 text-slate-400 mb-4" />
                                  <p className="text-slate-500">Enter a search query to find relevant documents</p>
                                </div>
                              ) : null}
                            </div>
                          </div>
                        </DialogContent>
                      </Dialog>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-red-400 hover:text-red-600"
                      >
                        <Trash2 size={16} />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <FileText className="mx-auto h-12 w-12 text-slate-400 mb-4" />
                <h3 className="text-lg font-medium text-slate-800 mb-2">No documents uploaded</h3>
                <p className="text-slate-600">Upload PDF documents to get started.</p>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
