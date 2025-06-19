import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Navigation, TopBar } from "@/components/navigation";
import Dashboard from "@/pages/dashboard";
import CreateAgent from "@/pages/create-agent";
import Chat from "@/pages/chat";
import Documents from "@/pages/documents";
import AgentDetails from "@/pages/agent-details";
import NotFound from "@/pages/not-found";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Dashboard} />
      <Route path="/create-agent" component={CreateAgent} />
      <Route path="/chat" component={Chat} />
      <Route path="/documents" component={Documents} />
      <Route path="/agents/:id" component={AgentDetails} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <div className="min-h-screen bg-slate-50">
          {/* Sidebar Navigation */}
          <Navigation />
          
          {/* Main Content Area */}
          <div className="lg:pl-64">
            <TopBar />
            <main>
              <Router />
            </main>
          </div>
        </div>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
