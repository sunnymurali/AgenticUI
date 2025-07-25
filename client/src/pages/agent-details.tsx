import { useState } from "react";
import React from "react";
import { useParams, Link } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useAgent, useUpdateAgent } from "@/hooks/use-agents";
import { Bot, MessageCircle, FileText, Edit, ArrowLeft, Thermometer, Save, X, Loader2, Wrench, Globe, Key, Code } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { insertAgentSchema } from "@shared/schema";
import type { InsertAgent } from "@shared/schema";

export default function AgentDetails() {
  const params = useParams();
  const agentId = params.id;
  const [isEditing, setIsEditing] = useState(false);
  
  const { data: agentData, isLoading } = useAgent(agentId!);
  const updateMutation = useUpdateAgent(agentId!);

  const form = useForm<InsertAgent>({
    resolver: zodResolver(insertAgentSchema),
    defaultValues: {
      name: "",
      model_name: "",
      system_prompt: "",
      temperature: 0.7,
      retriever_strategy: "",
    },
  });

  // Update form when agent data loads
  React.useEffect(() => {
    if (agentData?.agent) {
      form.reset({
        name: agentData.agent.name,
        model_name: agentData.agent.model_name,
        system_prompt: agentData.agent.system_prompt,
        temperature: agentData.agent.temperature,
        retriever_strategy: agentData.agent.retriever_strategy || "",
      });
    }
  }, [agentData, form]);

  const onSubmit = (data: InsertAgent) => {
    updateMutation.mutate(data, {
      onSuccess: () => {
        setIsEditing(false);
      },
    });
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    if (agentData?.agent) {
      form.reset({
        name: agentData.agent.name,
        model_name: agentData.agent.model_name,
        system_prompt: agentData.agent.system_prompt,
        temperature: agentData.agent.temperature,
        retriever_strategy: agentData.agent.retriever_strategy || "",
      });
    }
  };

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="max-w-4xl mx-auto space-y-6">
          <Skeleton className="h-8 w-64" />
          <Card>
            <CardHeader>
              <div className="flex items-center space-x-4">
                <Skeleton className="w-16 h-16 rounded-lg" />
                <div className="space-y-2">
                  <Skeleton className="h-6 w-48" />
                  <Skeleton className="h-4 w-32" />
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <Skeleton className="h-24 w-full" />
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  if (!agentData) {
    return (
      <div className="p-6">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-2xl font-semibold text-slate-800 mb-4">Agent not found</h1>
          <Link href="/">
            <Button>Back to Dashboard</Button>
          </Link>
        </div>
      </div>
    );
  }

  const { agent, documents, document_count } = agentData;

  return (
    <div className="p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Link href="/">
              <Button variant="ghost" size="sm">
                <ArrowLeft className="mr-2" size={16} />
                Back to Dashboard
              </Button>
            </Link>
            <h1 className="text-2xl font-semibold text-slate-800">Agent Details</h1>
          </div>
          <div className="flex space-x-3">
            <Link href={`/chat?agent=${agent.agent_id}`}>
              <Button className="bg-primary text-white hover:bg-blue-700">
                <MessageCircle className="mr-2" size={16} />
                Chat
              </Button>
            </Link>
            <Button variant="outline" onClick={() => setIsEditing(true)}>
              <Edit className="mr-2" size={16} />
              Edit
            </Button>
          </div>
        </div>

        {/* Agent Info Card */}
        {isEditing ? (
          <Card>
            <CardHeader>
              <CardTitle>Edit Agent</CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                <div>
                  <Label htmlFor="name">Agent Name</Label>
                  <Input
                    id="name"
                    {...form.register("name")}
                    placeholder="Enter agent name"
                  />
                  {form.formState.errors.name && (
                    <p className="text-red-500 text-sm mt-1">{form.formState.errors.name.message}</p>
                  )}
                </div>

                <div>
                  <Label htmlFor="model_name">Model</Label>
                  <Select
                    value={form.watch("model_name")}
                    onValueChange={(value) => form.setValue("model_name", value)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select a model" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="gpt-4">GPT-4</SelectItem>
                      <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo</SelectItem>
                      <SelectItem value="claude-3-sonnet">Claude 3 Sonnet</SelectItem>
                      <SelectItem value="claude-3-haiku">Claude 3 Haiku</SelectItem>
                    </SelectContent>
                  </Select>
                  {form.formState.errors.model_name && (
                    <p className="text-red-500 text-sm mt-1">{form.formState.errors.model_name.message}</p>
                  )}
                </div>

                <div>
                  <Label htmlFor="system_prompt">System Prompt</Label>
                  <Textarea
                    id="system_prompt"
                    {...form.register("system_prompt")}
                    placeholder="Enter system prompt"
                    rows={6}
                  />
                  {form.formState.errors.system_prompt && (
                    <p className="text-red-500 text-sm mt-1">{form.formState.errors.system_prompt.message}</p>
                  )}
                </div>

                <div>
                  <Label htmlFor="temperature">Temperature ({form.watch("temperature")})</Label>
                  <Input
                    id="temperature"
                    type="number"
                    min="0"
                    max="2"
                    step="0.1"
                    {...form.register("temperature", { valueAsNumber: true })}
                  />
                  {form.formState.errors.temperature && (
                    <p className="text-red-500 text-sm mt-1">{form.formState.errors.temperature.message}</p>
                  )}
                </div>

                <div>
                  <Label htmlFor="retriever_strategy">Retriever Strategy</Label>
                  <Input
                    id="retriever_strategy"
                    {...form.register("retriever_strategy")}
                    placeholder="Enter retriever strategy (optional)"
                  />
                </div>

                <div className="flex space-x-3 pt-4">
                  <Button 
                    type="submit" 
                    disabled={updateMutation.isPending}
                    className="bg-primary text-white hover:bg-blue-700"
                  >
                    {updateMutation.isPending ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Saving...
                      </>
                    ) : (
                      <>
                        <Save className="mr-2" size={16} />
                        Save Changes
                      </>
                    )}
                  </Button>
                  <Button 
                    type="button" 
                    variant="outline" 
                    onClick={handleCancelEdit}
                  >
                    <X className="mr-2" size={16} />
                    Cancel
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>
        ) : (
          <Card>
          <CardHeader>
            <div className="flex items-center space-x-4">
              <div className="w-16 h-16 gradient-blue-teal rounded-lg flex items-center justify-center">
                <Bot className="text-white text-2xl" />
              </div>
              <div className="flex-1">
                <CardTitle className="text-xl">{agent.name}</CardTitle>
                <p className="text-slate-600 mt-1">
                  Model: <Badge variant="secondary">{agent.model_name}</Badge>
                </p>
                <p className="text-sm text-slate-500 mt-2">
                  Created {new Date(agent.created_at).toLocaleDateString()}
                </p>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h3 className="font-medium text-slate-800 mb-2">System Prompt</h3>
              <div className="bg-slate-50 p-4 rounded-lg">
                <p className="text-slate-700 whitespace-pre-wrap">{agent.system_prompt}</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-slate-50 p-4 rounded-lg">
                <div className="flex items-center space-x-2 mb-2">
                  <Thermometer size={16} className="text-slate-600" />
                  <span className="font-medium text-slate-800">Temperature</span>
                </div>
                <p className="text-2xl font-semibold text-slate-800">{agent.temperature}</p>
                <p className="text-xs text-slate-500">
                  {agent.temperature < 0.3 ? 'Focused' : agent.temperature > 0.7 ? 'Creative' : 'Balanced'}
                </p>
              </div>

              <div className="bg-slate-50 p-4 rounded-lg">
                <div className="flex items-center space-x-2 mb-2">
                  <FileText size={16} className="text-slate-600" />
                  <span className="font-medium text-slate-800">Documents</span>
                </div>
                <p className="text-2xl font-semibold text-slate-800">{document_count}</p>
                <p className="text-xs text-slate-500">Uploaded files</p>
              </div>

              <div className="bg-slate-50 p-4 rounded-lg">
                <div className="flex items-center space-x-2 mb-2">
                  <MessageCircle size={16} className="text-slate-600" />
                  <span className="font-medium text-slate-800">Interactions</span>
                </div>
                <p className="text-2xl font-semibold text-slate-800">{agent.interactions?.length || 0}</p>
                <p className="text-xs text-slate-500">Total conversations</p>
              </div>
            </div>

            {agent.retriever_strategy && (
              <div>
                <h3 className="font-medium text-slate-800 mb-2">Retriever Strategy</h3>
                <Badge variant="outline">{agent.retriever_strategy}</Badge>
              </div>
            )}
          </CardContent>
        </Card>
        )}
        {/* Tool Information - Show when tool_id is present */}
        {/* Tools Information - Show when tools array has items */}
        {agent.tools && agent.tools.length > 0 && (
          <Card>
            <CardHeader>
              <div className="flex items-center space-x-2">
                <Wrench className="text-amber-600" size={20} />
                <CardTitle>Tools Configuration</CardTitle>
                <Badge variant="default" className="bg-amber-100 text-amber-800">
                  {agent.tools.length} Tool{agent.tools.length > 1 ? 's' : ''} Enabled
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              {agent.tools.map((tool: any, index: number) => (
                <div key={tool.tool_id || index} className="border rounded-lg p-4 bg-white">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-medium text-slate-800 flex items-center space-x-2">
                      <Code size={16} className="text-amber-600" />
                      <span>{tool.tool_name || `Tool ${index + 1}`}</span>
                    </h3>
                    <Badge variant="outline" className="text-xs">
                      Tool #{index + 1}
                    </Badge>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    <div className="bg-slate-50 p-3 rounded-lg">
                      <div className="flex items-center space-x-2 mb-2">
                        <Wrench size={14} className="text-slate-600" />
                        <span className="font-medium text-slate-800 text-sm">Tool ID</span>
                      </div>
                      <p className="text-xs font-mono text-slate-700 bg-white p-2 rounded border">
                        {tool.tool_id || 'N/A'}
                      </p>
                    </div>
                    <div className="bg-slate-50 p-3 rounded-lg">
                      <div className="flex items-center space-x-2 mb-2">
                        <Globe size={14} className="text-slate-600" />
                        <span className="font-medium text-slate-800 text-sm">Endpoint URL</span>
                      </div>
                      <p className="text-xs font-mono text-slate-700 bg-white p-2 rounded border break-all">
                        {tool.endpoint_url || 'N/A'}
                      </p>
                    </div>
                    <div className="bg-slate-50 p-3 rounded-lg">
                      <div className="flex items-center space-x-2 mb-2">
                        <Key size={14} className="text-slate-600" />
                        <span className="font-medium text-slate-800 text-sm">API Token</span>
                      </div>
                      <p className="text-xs font-mono text-slate-700 bg-white p-2 rounded border">
                        {tool.api_token ? '••••••••••••••••' : 'N/A'}
                      </p>
                    </div>
                    <div className="bg-slate-50 p-3 rounded-lg">
                      <div className="flex items-center space-x-2 mb-2">
                        <span className="font-medium text-slate-800 text-sm">Created</span>
                      </div>
                      <p className="text-xs text-slate-700">
                        {tool.created_at ? new Date(tool.created_at).toLocaleDateString() : 'N/A'}
                      </p>
                    </div>
                  </div>
                  {tool.tool_description && (
                    <div className="mb-4">
                      <h4 className="font-medium text-slate-800 mb-2 text-sm">Description</h4>
                      <div className="bg-slate-50 p-3 rounded-lg">
                        <p className="text-slate-700 text-sm">{tool.tool_description}</p>
                      </div>
                    </div>
                  )}
                  {tool.tool_parameters && (
                    <div>
                      <h4 className="font-medium text-slate-800 mb-2 text-sm">Parameters</h4>
                      <div className="bg-slate-50 p-3 rounded-lg">
                        {(() => {
                          try {
                            const params = JSON.parse(tool.tool_parameters);
                            return (
                              <div className="space-y-2">
                                {Object.entries(params).map(([key, value]) => (
                                  <div key={key} className="bg-white p-2 rounded border">
                                    <div className="flex items-center justify-between mb-1">
                                      <span className="font-medium text-slate-800 capitalize text-sm">{key.replace('_', ' ')}</span>
                                      <Badge variant="outline" className="text-xs">
                                        {Array.isArray(value) ? 'List' : typeof value}
                                      </Badge>
                                    </div>
                                    {Array.isArray(value) ? (
                                      <div className="space-y-1">
                                        {value.map((item, idx) => (
                                          <div key={idx} className="bg-slate-50 p-1 rounded text-xs">
                                            {typeof item === 'object' ? (
                                              <pre className="text-xs text-slate-600 overflow-x-auto">
                                                {JSON.stringify(item, null, 2)}
                                              </pre>
                                            ) : (
                                              <span className="text-slate-700">{String(item)}</span>
                                            )}
                                          </div>
                                        ))}
                                      </div>
                                    ) : typeof value === 'object' ? (
                                      <pre className="text-xs text-slate-700 bg-slate-50 p-1 rounded overflow-x-auto">
                                        {JSON.stringify(value, null, 2)}
                                      </pre>
                                    ) : (
                                      <span className="text-xs text-slate-700">{String(value)}</span>
                                    )}
                                  </div>
                                ))}
                              </div>
                            );
                          } catch (error) {
                            return (
                              <div className="bg-red-50 p-2 rounded border border-red-200">
                                <p className="text-red-600 text-xs">Invalid JSON format</p>
                                <pre className="text-xs text-red-500 mt-1 overflow-x-auto">
                                  {tool.tool_parameters}
                                </pre>
                              </div>
                            );
                          }
                        })()}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </CardContent>
          </Card>
        )}
        {/* Documents */}
        {documents.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Uploaded Documents</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {documents.map((filename, index) => (
                  <div
                    key={index}
                    className="flex items-center space-x-3 p-3 border border-gray-200 rounded-lg"
                  >
                    <div className="w-10 h-10 bg-red-100 rounded-lg flex items-center justify-center">
                      <FileText className="text-red-600" size={20} />
                    </div>
                    <div className="flex-1">
                      <p className="font-medium text-slate-800">{filename}</p>
                      <p className="text-sm text-slate-500">PDF Document</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Recent Interactions */}
        {agent.interactions && agent.interactions.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Recent Interactions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {agent.interactions.slice(0, 5).map((interaction, index) => (
                  <div key={index} className="border-l-4 border-primary pl-4">
                    <div className="space-y-2">
                      <div>
                        <p className="text-sm font-medium text-slate-800">User:</p>
                        <p className="text-sm text-slate-600">{interaction.user}</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium text-slate-800">Assistant:</p>
                        <p className="text-sm text-slate-600">{interaction.assistant}</p>
                      </div>
                      <p className="text-xs text-slate-500">
                        {new Date(interaction.timestamp).toLocaleString()}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
