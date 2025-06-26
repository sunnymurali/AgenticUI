import { useState, useEffect, useRef } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ChatMessageComponent } from "@/components/chat-message";
import { useAgents } from "@/hooks/use-agents";
import { useChat } from "@/hooks/use-chat";
import { useLocation } from "wouter";
// ADDED: Wrench icon import
import { Bot, Send, Plus, Paperclip, FileText, X, Wrench } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";

export default function Chat() {
  const [location] = useLocation();
  const [selectedAgentId, setSelectedAgentId] = useState<string>("");
  const [currentMessage, setCurrentMessage] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const { data: agents, isLoading: agentsLoading } = useAgents();
  const [showDocumentPicker, setShowDocumentPicker] = useState(false);
  const [availableDocuments, setAvailableDocuments] = useState<string[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null);
  const [loadingDocuments, setLoadingDocuments] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [fileNames, setFileNames] = useState<string[]>([]);

  // Added tool status detection
  const selectedAgent = agents?.find(agent => agent.agent_id === selectedAgentId);
  const useTools = selectedAgent?.use_tools || false;
  const chat = useChat(selectedAgentId);
  // Parse agent from URL
  useEffect(() => {
    const params = new URLSearchParams(location.split('?')[1] || '');
    const agentParam = params.get('agent');
    if (agentParam && agents) {
      const agent = agents.find(a => a.agent_id === agentParam);
      if (agent) {
        setSelectedAgentId(agentParam);
      }
    }
  }, [location, agents]);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chat.messages]);

  const handleMessageChange = async (value: string) => {
    setCurrentMessage(value);
    
    // Check if user typed "/" to trigger document picker
    if (value === "/" && selectedAgentId) {
      setLoadingDocuments(true);
      try {
        const documents = await api.getDocumentsByFileName(selectedAgentId);
        setAvailableDocuments(documents);
        setShowDocumentPicker(true);
      } catch (error) {
        console.error("Failed to load documents:", error);
      } finally {
        setLoadingDocuments(false);
      }
    } else if (showDocumentPicker && !value.startsWith("/")) {
      setShowDocumentPicker(false);
      setSelectedDocument(null);
    }
  };
  
  const handleDocumentSelect = (docName: string) => {
    setSelectedDocument(docName);
    setCurrentMessage("");
    setShowDocumentPicker(false);
    textareaRef.current?.focus();
  };
  
  const handleRemoveDocument = () => {
    setSelectedDocument(null);
  };

  const handleSendMessage = (e: React.FormEvent) => {
  e.preventDefault();
  if (!currentMessage.trim() || !selectedAgentId || chat.isLoading) return;
  
  chat.sendMessage(currentMessage.trim(), selectedDocument || undefined);
  setCurrentMessage("");
  setSelectedDocument(null);
  setShowDocumentPicker(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e);
    }
  };

  const handleNewSession = () => {
    chat.clearChat();
  };

  return (
    <div className="p-6 h-[calc(100vh-8rem)]">
      <div className="flex h-full max-h-full">
        {/* Agent Selection Sidebar */}
        <div className="w-80 bg-white border border-gray-200 rounded-l-lg flex flex-col">
          <div className="p-4 border-b border-gray-200">
            <h3 className="font-semibold text-slate-800 mb-3">Select Agent</h3>
            <div className="space-y-2">
              {agentsLoading ? (
                [...Array(3)].map((_, i) => (
                  <div key={i} className="p-3 border border-gray-200 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <Skeleton className="w-8 h-8 rounded-lg" />
                      <div className="flex-1">
                        <Skeleton className="h-4 w-24 mb-1" />
                        <Skeleton className="h-3 w-16" />
                      </div>
                    </div>
                  </div>
                ))
              ) : agents && agents.length > 0 ? (
                agents.map((agent) => (
                  <div
                    key={agent.agent_id}
                    className={`p-3 border rounded-lg cursor-pointer transition-colors duration-200 ${
                      selectedAgentId === agent.agent_id
                        ? 'border-primary bg-blue-50'
                        : 'border-gray-200 hover:bg-slate-50'
                    }`}
                    onClick={() => setSelectedAgentId(agent.agent_id)}
                  >
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 gradient-blue-teal rounded-lg flex items-center justify-center">
                        <Bot className="text-white text-sm" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-slate-800 truncate">{agent.name}</p>
                        <p className="text-xs text-slate-500">{agent.model_name}</p>
                      </div>
                      {agent.use_tools && (
                        <div className="flex items-center gap-1 mt-1">
                          <Wrench className="w-3 h-3 text-amber-600" />
                          <span className="text-xs text-amber-600">Tools</span>
                        </div>
                      )}  
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-4">
                  <p className="text-slate-500 text-sm">No agents available</p>
                </div>
              )}
            </div>
          </div>
          
          {/* Recent Sessions */}
          <div className="flex-1 p-4 overflow-y-auto">
            <h4 className="font-medium text-slate-700 mb-3">Recent Sessions</h4>
            <div className="text-center py-4">
              <p className="text-slate-500 text-sm">No recent sessions</p>
            </div>
          </div>
        </div>

        {/* Chat Interface */}
        <div className="flex-1 flex flex-col bg-white border-r border-t border-b border-gray-200 rounded-r-lg">
          {/* Chat Header */}
          <div className="border-b border-gray-200 p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 gradient-blue-teal rounded-lg flex items-center justify-center">
                  <Bot className="text-white" />
                </div>
                {selectedAgent?.use_tools && (
                  <div className="flex items-center gap-1 mt-1">
                    <Wrench className="w-3 h-3 text-amber-600" />
                    <span className="text-xs text-amber-600 font-medium">Tools Enabled</span>
                  </div>
                )}  
                <div>
                  <h3 className="font-semibold text-slate-800">
                    {selectedAgent ? selectedAgent.name : "Select an agent to start chatting"}
                  </h3>
                  <p className="text-sm text-slate-500">
                    {selectedAgent ? "Ready to help" : "Choose from the sidebar"}
                  </p>
                </div>
              </div>
              {selectedAgent && (
                <Button 
                  variant="outline"
                  onClick={handleNewSession}
                  className="text-sm"
                >
                  <Plus className="mr-2" size={16} />
                  New Session
                </Button>
              )}
            </div>
          </div>

          {/* Chat Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50">
            {selectedAgent ? (
              <>
                {/* Welcome Message */}
                <div className="chat-message">
                  <div className="w-8 h-8 gradient-blue-teal rounded-full flex items-center justify-center flex-shrink-0">
                    <Bot className="text-white text-sm" />
                  </div>
                  <div className="message-bubble assistant">
                    <p className="text-slate-800">
                      Hello! I'm {selectedAgent.name}. I'm ready to help you. What would you like to know?
                    </p>
                    <p className="text-xs text-slate-500 mt-2">Just now</p>
                  </div>
                </div>

                {/* Chat Messages */}
                {chat.messages.map((message, index) => (
                  <ChatMessageComponent
                    key={index}
                    message={message}
                    agentName={selectedAgent.name}
                  />
                ))}

                {/* Loading indicator */}
                {chat.isLoading && (
                  <div className="chat-message">
                    <div className="w-8 h-8 gradient-blue-teal rounded-full flex items-center justify-center flex-shrink-0">
                      <Bot className="text-white text-sm animate-pulse" />
                    </div>
                    <div className="message-bubble assistant">
                      <p className="text-slate-800">Thinking...</p>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center">
                <div className="text-center">
                  <Bot className="mx-auto h-12 w-12 text-slate-400 mb-4" />
                  <h3 className="text-lg font-medium text-slate-800 mb-2">No agent selected</h3>
                  <p className="text-slate-600">Choose an agent from the sidebar to start chatting.</p>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Chat Input */}
          {/* Chat Input */}
          {selectedAgent && (
            <div className="border-t border-gray-200 p-4">
              <div className="space-y-2">
                {/* Selected Document Display */}
                {selectedDocument && (
                  <div className="flex items-center gap-2 p-2 bg-blue-50 rounded-lg border border-blue-200">
                    <FileText size={16} className="text-blue-600" />
                    <span className="text-sm text-blue-800 font-medium">{selectedDocument}</span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleRemoveDocument}
                      className="ml-auto h-6 w-6 p-0 text-blue-600 hover:text-blue-800"
                    >
                      <X size={14} />
                    </Button>
                  </div>
                )}
                {/* Document Picker */}
                {showDocumentPicker && (
                  <div className="border border-gray-200 rounded-lg bg-white shadow-sm max-h-48 overflow-y-auto">
                    <div className="p-2 border-b border-gray-100 bg-gray-50">
                      <span className="text-sm font-medium text-gray-700">Select a document:</span>
                    </div>
                    {loadingDocuments ? (
                      <div className="p-4 text-center">
                        <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto" />
                        <span className="text-sm text-gray-500 mt-2">Loading documents...</span>
                      </div>
                    ) : availableDocuments.length > 0 ? (
                      <div className="py-1">
                        {availableDocuments.map((doc, index) => (
                          <button
                            key={index}
                            onClick={() => handleDocumentSelect(doc)}
                            className="w-full text-left px-3 py-2 hover:bg-gray-50 flex items-center gap-2 text-sm"
                          >
                            <FileText size={14} className="text-gray-400" />
                            <span>{doc}</span>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <div className="p-4 text-center text-sm text-gray-500">
                        No documents available
                      </div>
                    )}
                  </div>
                )}
                <form onSubmit={handleSendMessage} className="flex items-end space-x-3">
                  <div className="flex-1">
                    <div className="relative">
                      <Textarea
                        ref={textareaRef}
                        value={currentMessage}
                        onChange={(e) => handleMessageChange(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder={selectedDocument ? "Ask about the selected document..." : "Type your message... (Type '/' to select a document)"}
                        rows={1}
                        className="resize-none pr-10"
                        disabled={chat.isLoading}
                      />
                      {currentMessage === "/" && (
                        <div className="absolute right-2 top-2 text-blue-500">
                          <FileText size={16} />
                        </div>
                      )}
                    </div>
                  </div>
                  <Button
                    type="submit"
                    className="bg-primary text-white hover:bg-blue-700"
                    disabled={!currentMessage.trim() || chat.isLoading}
                  >
                    {chat.isLoading ? (
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <Send size={16} />
                    )}
                  </Button>
                </form>
              </div>
              <p className="text-xs text-slate-500 mt-2">
                Press Enter to send, Shift+Enter for new line • Type "/" to select a document
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
