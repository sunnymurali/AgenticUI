import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";
import type { CheckedState } from "@radix-ui/react-checkbox";

import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { insertAgentSchema, toolDefinitionSchema } from "@shared/schema";
import type { InsertAgent, ToolDefinition } from "@shared/schema";
import { useCreateAgent } from "@/hooks/use-agents";
import { useLocation } from "wouter";
import { Plus, Trash2, Wrench, Loader2 } from "lucide-react";

import { api } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

export default function CreateAgent() {
  const [, setLocation] = useLocation();
  // NEW: Tool state management
  const [enableTools, setEnableTools] = useState(false);
  const [tools, setTools] = useState<ToolDefinition[]>([]);
  const createAgentMutation = useCreateAgent();
  const { toast } = useToast();

  const form = useForm<InsertAgent>({
    resolver: zodResolver(insertAgentSchema),
    defaultValues: {
      name: "",
      model_name: "gpt-4o",
      system_prompt: "",
      temperature: 0.7,
      retriever_strategy: "none",
    },
    mode: "onChange"  // <-- ADD THIS LINE
  });

  // NEW: Tool form setup
  const toolForm = useForm<ToolDefinition>({
    resolver: zodResolver(toolDefinitionSchema),
    defaultValues: {
      name: "",
      description: "",
      endpoint_url: "",
      api_token: "",
    },
  });

  // Force form values to never be empty strings
  React.useEffect(() => {
    const subscription = form.watch((value, { name }) => {
      if (name === 'model_name' && (!value.model_name || value.model_name === '')) {
        form.setValue('model_name', 'gpt-4o');
      }
      if (name === 'retriever_strategy' && (!value.retriever_strategy || value.retriever_strategy === '')) {
        form.setValue('retriever_strategy', 'none');
      }
    });
    return () => subscription.unsubscribe();
  }, [form]);

  // NEW: Tool management functions
  const addTool = (toolData: ToolDefinition) => {
    setTools(prev => [...prev, toolData]);
    toolForm.reset();
  };

  const removeTool = (index: number) => {
    setTools(prev => prev.filter((_, i) => i !== index));
  };

  const onSubmit = async (data: InsertAgent) => {
    try {
      // Ensure no empty string values
      const cleanedData = {
        ...data,
        model_name: data.model_name || "gpt-4o",
        retriever_strategy: data.retriever_strategy || "none"
      };
      // NEW: Add use_tools flag to agent data
      // Add use_tools flag to agent data
      const agentData = {
        ...cleanedData,
        use_tools: enableTools && tools.length > 0
      };
      
      const agent = await createAgentMutation.mutateAsync(agentData);
      
      // NEW: Create tools for the agent
      if (enableTools && tools.length > 0) {
        for (const tool of tools) {
          try {
            await api.createTool(agent.agent_id, tool);
          } catch (error) {
            toast({
              title: "Warning",
              description: `Failed to create tool "${tool.name}": ${error instanceof Error ? error.message : 'Unknown error'}`,
              variant: "destructive",
            });
          }
        }
      }
      
      setLocation("/");
    } catch (error) {
      // Error is handled by the mutation
    }
  };

  const temperatureValue = form.watch("temperature");

  return (
    <div className="p-6">
      <div className="max-w-2xl mx-auto">
        <Card>
          <CardHeader>
            <CardTitle className="text-2xl font-semibold text-slate-800">
              Create New Agent
            </CardTitle>
            <p className="text-slate-600">
              Configure your AI agent with custom settings and capabilities.
            </p>
          </CardHeader>
          <CardContent>
            <Form {...form}>
              <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
                <FormField
                  control={form.control}
                  name="name"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Agent Name</FormLabel>
                      <FormControl>
                        <Input 
                          placeholder="Enter agent name"
                          {...field}
                        />
                      </FormControl>
                      <FormDescription>
                        Give your agent a descriptive name
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="model_name"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Model Name</FormLabel>
                      <Select 
                        onValueChange={(value) => field.onChange(value || "gpt-4o")} 
                        value={field.value || "gpt-4o"}
                      > 
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select a model" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="gpt-4o">GPT-4o</SelectItem>
                          <SelectItem value="gpt-4o-mini">GPT-4o Mini</SelectItem>
                          <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="system_prompt"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>System Prompt</FormLabel>
                      <FormControl>
                        <Textarea
                          placeholder="Define the agent's behavior and role..."
                          className="min-h-[120px]"
                          {...field}
                        />
                      </FormControl>
                      <FormDescription>
                        This defines how your agent will behave and respond to queries.
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="temperature"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Temperature ({temperatureValue})</FormLabel>
                      <FormControl>
                        <Slider
                          min={0}
                          max={2}
                          step={0.1}
                          value={[field.value]}
                          onValueChange={(vals) => field.onChange(vals[0])}
                          className="w-full"
                        />
                      </FormControl>
                      <FormDescription>
                        Controls randomness in responses. Lower values make responses more focused and deterministic.
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="retriever_strategy"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Retrieval Strategy</FormLabel>
                      <FormControl>
                        <Select 
                            onValueChange={(value) => field.onChange(value || "none")}
                            value={field.value || "none"}>
                          <SelectTrigger>
                            <SelectValue placeholder="Select retrieval strategy" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="none">None</SelectItem>
                            <SelectItem value="vector_rag">Vector RAG</SelectItem>
                            <SelectItem value="semantic_search">Semantic Search</SelectItem>
                          </SelectContent>
                        </Select>
                      </FormControl>
                      <FormDescription>
                        Choose how the agent retrieves relevant information from documents.
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                {/* Tools Section */}
                <div className="space-y-4 border-t pt-6">
                  <div className="flex items-center space-x-2">
                    <Checkbox 
                      id="enable-tools" 
                      checked={enableTools}
                      onCheckedChange={(checked: CheckedState) => setEnableTools(checked === true)}
                    />
                    <label 
                      htmlFor="enable-tools" 
                      className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                    >
                      Enable Tools
                    </label>
                  </div>
                  
                  {enableTools && (
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <Wrench className="h-5 w-5" />
                          Agent Tools
                        </CardTitle>
                        <p className="text-sm text-muted-foreground">
                          Add REST API tools that your agent can use during conversations.
                        </p>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        {/* Existing Tools List */}
                        {tools.length > 0 && (
                          <div className="space-y-2">
                            <h4 className="text-sm font-medium">Configured Tools</h4>
                            {tools.map((tool, index) => (
                              <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                                <div>
                                  <p className="font-medium">{tool.name}</p>
                                  <p className="text-sm text-muted-foreground">{tool.description}</p>
                                  <p className="text-xs text-muted-foreground">{tool.endpoint_url}</p>
                                </div>
                                <Button
                                  type="button"
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => removeTool(index)}
                                >
                                  <Trash2 className="h-4 w-4" />
                                </Button>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Add New Tool Form */}
                        <Form {...toolForm}>
                          <form onSubmit={toolForm.handleSubmit(addTool)} className="space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                              <FormField
                                control={toolForm.control}
                                name="name"
                                render={({ field }) => (
                                  <FormItem>
                                    <FormLabel>Tool Name</FormLabel>
                                    <FormControl>
                                      <Input placeholder="e.g., Weather API" {...field} />
                                    </FormControl>
                                    <FormMessage />
                                  </FormItem>
                                )}
                              />
                              
                              <FormField
                                control={toolForm.control}
                                name="endpoint_url"
                                render={({ field }) => (
                                  <FormItem>
                                    <FormLabel>Endpoint URL</FormLabel>
                                    <FormControl>
                                      <Input placeholder="https://api.example.com/endpoint" {...field} />
                                    </FormControl>
                                    <FormMessage />
                                  </FormItem>
                                )}
                              />
                            </div>

                            <FormField
                              control={toolForm.control}
                              name="description"
                              render={({ field }) => (
                                <FormItem>
                                  <FormLabel>Tool Description</FormLabel>
                                  <FormControl>
                                    <Textarea 
                                      placeholder="Describe what this tool does and when to use it..."
                                      {...field} 
                                    />
                                  </FormControl>
                                  <FormMessage />
                                </FormItem>
                              )}
                            />

                            <FormField
                              control={toolForm.control}
                              name="api_token"
                              render={({ field }) => (
                                <FormItem>
                                  <FormLabel>API Token</FormLabel>
                                  <FormControl>
                                    <Input 
                                      type="password"
                                      placeholder="Enter API token or key..."
                                      {...field} 
                                    />
                                  </FormControl>
                                  <FormMessage />
                                </FormItem>
                              )}
                            />

                            <Button type="submit" variant="outline" className="w-full">
                              <Plus className="mr-2 h-4 w-4" />
                              Add Tool
                            </Button>
                          </form>
                        </Form>
                      </CardContent>
                    </Card>
                  )}
                </div>

                <div className="flex gap-4">
                  <Button 
                    type="submit" 
                    disabled={createAgentMutation.isPending}
                    className="flex-1"
                  >
                    {createAgentMutation.isPending && (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    )}
                    Create Agent
                  </Button>
                  <Button 
                    type="button" 
                    variant="outline" 
                    onClick={() => setLocation("/")}
                    className="flex-1"
                  >
                    Cancel
                  </Button>
                </div>
              </form>
            </Form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}