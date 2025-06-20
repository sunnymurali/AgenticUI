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
import { Bot, MessageCircle, FileText, Edit, ArrowLeft, Thermometer, Save, X, Loader2 } from "lucide-react";
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
            <Button variant="outline">
              <Edit className="mr-2" size={16} />
              Edit
            </Button>
          </div>
        </div>

        {/* Agent Info Card */}
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
