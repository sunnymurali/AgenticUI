import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Bot, MessageCircle, Edit, Trash2, Code, BarChart3 } from "lucide-react";
import type { AgentWithStats } from "@/lib/types";
import { Link } from "wouter";

interface AgentCardProps {
  agent: AgentWithStats;
  onDelete: (agentId: string) => void;
  onEdit: (agentId: string) => void;
}

const getAgentIcon = (name: string | undefined | null) => {
  if (!name || typeof name !== 'string') {
    return Bot;
  }

  const lowerName = name.toLowerCase();

  if (lowerName.includes('code') || lowerName.includes('programming')) {
    return Code;
  }
  if (lowerName.includes('data') || lowerName.includes('analyst')) {
    return BarChart3;
  }

  return Bot;
};

const getGradientClass = (index: number) => {
  const gradients = [
    "gradient-blue-teal",
    "gradient-green-blue", 
    "gradient-purple-pink"
  ];
  return gradients[index % gradients.length];
};

export function AgentCard({ agent, onDelete, onEdit }: AgentCardProps) {
  const Icon = getAgentIcon(agent.name);
  const gradientClass = getGradientClass(
    agent?.agent_id ? parseInt(agent.agent_id.slice(-1), 16) : 0
  );

  const handleDelete = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (confirm('Are you sure you want to delete this agent?')) {
      onDelete(agent.agent_id);
    }
  };

  const handleEdit = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onEdit(agent.agent_id);
  };

  return (
    <Card className="agent-card">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center space-x-3">
          <div className={`w-10 h-10 ${gradientClass} rounded-lg flex items-center justify-center`}>
            <Icon className="text-white" size={20} />
          </div>
          <div>
            <h4 className="font-medium text-slate-800">{agent.name}</h4>
            <p className="text-xs text-slate-500">{agent.model_name}</p>
          </div>
        </div>
        <div className="flex items-center space-x-1">
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={handleEdit}
            className="p-1 hover:bg-slate-100"
          >
            <Edit className="text-slate-400 text-sm" size={14} />
          </Button>
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={handleDelete}
            className="p-1 hover:bg-red-100"
          >
            <Trash2 className="text-red-400 text-sm" size={14} />
          </Button>
        </div>
      </div>
      
      <p className="text-sm text-slate-600 mb-3 line-clamp-2">
        {agent.system_prompt?.substring(0, 100) ?? 'No prompt available'}...
      </p>
      
      <div className="flex items-center justify-between text-xs text-slate-500 mb-3">
        <span>Temperature: <span className="font-medium">{agent.temperature}</span></span>
        <span><span className="font-medium">{agent.document_count}</span> docs</span>
      </div>
      
      <div className="flex space-x-2">
        <Link href={`/chat?agent=${agent.agent_id}`}>
          <Button className="flex-1 bg-primary text-white hover:bg-blue-700 text-xs">
            <MessageCircle className="mr-1" size={12} />
            Chat
          </Button>
        </Link>
        <Link href={`/agents/${agent.agent_id}`}>
          <Button variant="secondary" className="flex-1 text-xs">
            View
          </Button>
        </Link>
      </div>
    </Card>
  );
}
