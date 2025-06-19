import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AgentCard } from "@/components/agent-card";
import { useAgents, useDeleteAgent, useStats } from "@/hooks/use-agents";
import { Link } from "wouter";
import { Plus, Bot, MessageCircle, FileText, Search } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";

export default function Dashboard() {
  const { data: agents, isLoading: agentsLoading } = useAgents();
  const { data: stats, isLoading: statsLoading } = useStats();
  const deleteAgentMutation = useDeleteAgent();

  const handleDeleteAgent = (agentId: string) => {
    deleteAgentMutation.mutate(agentId);
  };

  const handleEditAgent = (agentId: string) => {
    // TODO: Implement edit functionality
    console.log("Edit agent:", agentId);
  };

  if (statsLoading) {
    return (
      <div className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {[...Array(4)].map((_, i) => (
            <Card key={i}>
              <CardContent className="p-6">
                <Skeleton className="h-4 w-24 mb-2" />
                <Skeleton className="h-8 w-12" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="p-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <Card className="stats-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-600">Total Agents</p>
              <p className="text-2xl font-semibold text-slate-800">
                {stats?.totalAgents || 0}
              </p>
            </div>
            <div className="stats-icon bg-blue-100">
              <Bot className="text-blue-600" />
            </div>
          </div>
        </Card>

        <Card className="stats-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-600">Active Sessions</p>
              <p className="text-2xl font-semibold text-slate-800">
                {stats?.totalSessions || 0}
              </p>
            </div>
            <div className="stats-icon bg-green-100">
              <MessageCircle className="text-green-600" />
            </div>
          </div>
        </Card>

        <Card className="stats-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-600">Documents</p>
              <p className="text-2xl font-semibold text-slate-800">
                {stats?.totalDocuments || 0}
              </p>
            </div>
            <div className="stats-icon bg-purple-100">
              <FileText className="text-purple-600" />
            </div>
          </div>
        </Card>

        <Card className="stats-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-600">Total Queries</p>
              <p className="text-2xl font-semibold text-slate-800">
                {stats?.totalQueries || 0}
              </p>
            </div>
            <div className="stats-icon bg-orange-100">
              <Search className="text-orange-600" />
            </div>
          </div>
        </Card>
      </div>

      {/* Recent Agents */}
      <Card>
        <CardHeader className="border-b border-gray-200">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg font-semibold text-slate-800">
              Your AI Agents
            </CardTitle>
            <Link href="/create-agent">
              <Button className="bg-primary text-white hover:bg-blue-700">
                <Plus className="mr-2" size={16} />
                Create Agent
              </Button>
            </Link>
          </div>
        </CardHeader>
        <CardContent className="p-6">
          {agentsLoading ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
              {[...Array(6)].map((_, i) => (
                <Card key={i} className="p-4">
                  <Skeleton className="h-10 w-10 rounded-lg mb-3" />
                  <Skeleton className="h-4 w-32 mb-2" />
                  <Skeleton className="h-3 w-24 mb-3" />
                  <Skeleton className="h-16 w-full mb-3" />
                  <div className="flex space-x-2">
                    <Skeleton className="h-8 flex-1" />
                    <Skeleton className="h-8 flex-1" />
                  </div>
                </Card>
              ))}
            </div>
          ) : agents && agents.length > 0 ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
              {agents.map((agent) => (
                <AgentCard
                  key={agent.agent_id}
                  agent={agent}
                  onDelete={handleDeleteAgent}
                  onEdit={handleEditAgent}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <Bot className="mx-auto h-12 w-12 text-slate-400 mb-4" />
              <h3 className="text-lg font-medium text-slate-800 mb-2">No agents yet</h3>
              <p className="text-slate-600 mb-4">Create your first AI agent to get started.</p>
              <Link href="/create-agent">
                <Button className="bg-primary text-white hover:bg-blue-700">
                  <Plus className="mr-2" size={16} />
                  Create Your First Agent
                </Button>
              </Link>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
