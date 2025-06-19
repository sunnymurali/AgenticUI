import { Bot, User } from "lucide-react";
import type { ChatMessage } from "@/lib/types";

interface ChatMessageProps {
  message: ChatMessage;
  agentName?: string;
}

export function ChatMessageComponent({ message, agentName }: ChatMessageProps) {
  const isUser = message.role === "user";
  const timestamp = message.timestamp ? new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : "Just now";

  if (isUser) {
    return (
      <div className="chat-message user">
        <div className="message-bubble user">
          <p>{message.content}</p>
          <p className="text-xs text-blue-200 mt-2">{timestamp}</p>
        </div>
        <div className="w-8 h-8 bg-slate-300 rounded-full flex items-center justify-center flex-shrink-0">
          <User className="text-slate-600 text-sm" />
        </div>
      </div>
    );
  }

  return (
    <div className="chat-message">
      <div className="w-8 h-8 gradient-blue-teal rounded-full flex items-center justify-center flex-shrink-0">
        <Bot className="text-white text-sm" />
      </div>
      <div className="message-bubble assistant max-w-2xl">
        <p className="text-slate-800 whitespace-pre-wrap">{message.content}</p>
        <p className="text-xs text-slate-500 mt-2">{timestamp}</p>
      </div>
    </div>
  );
}
