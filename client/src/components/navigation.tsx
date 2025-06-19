import { Link, useLocation } from "wouter";
import { Home, Plus, MessageCircle, FileText, Bot, Bell, User, Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { useState } from "react";

const navigation = [
  { name: "Dashboard", href: "/", icon: Home },
  { name: "Create Agent", href: "/create-agent", icon: Plus },
  { name: "Chat", href: "/chat", icon: MessageCircle },
  { name: "Documents", href: "/documents", icon: FileText },
];

export function Navigation() {
  const [location] = useLocation();
  const [open, setOpen] = useState(false);

  const NavContent = () => (
    <>
      <div className="p-6">
        <div className="flex items-center space-x-3 mb-8">
          <div className="w-10 h-10 gradient-blue-teal rounded-lg flex items-center justify-center">
            <Bot className="text-white text-lg" />
          </div>
          <h1 className="text-xl font-semibold text-slate-800">AI Agents</h1>
        </div>
        
        <nav className="space-y-2">
          {navigation.map((item) => {
            const isActive = location === item.href;
            const Icon = item.icon;
            
            return (
              <Link key={item.name} href={item.href}>
                <div
                  className={`nav-link ${isActive ? 'active' : ''}`}
                  onClick={() => setOpen(false)}
                >
                  <Icon className="w-5 h-5" />
                  <span>{item.name}</span>
                </div>
              </Link>
            );
          })}
        </nav>
      </div>
    </>
  );

  return (
    <>
      {/* Desktop Sidebar */}
      <div className="hidden lg:flex lg:w-64 lg:flex-col lg:fixed lg:inset-y-0">
        <div className="bg-white border-r border-gray-200 flex-1 flex flex-col">
          <NavContent />
        </div>
      </div>

      {/* Mobile Navigation */}
      <div className="lg:hidden">
        <Sheet open={open} onOpenChange={setOpen}>
          <SheetTrigger asChild>
            <Button variant="ghost" size="icon" className="lg:hidden">
              <Menu className="h-6 w-6" />
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="w-64 p-0">
            <div className="bg-white h-full">
              <NavContent />
            </div>
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}

export function TopBar() {
  const [location] = useLocation();
  const [open, setOpen] = useState(false);
  
  const getPageTitle = () => {
    switch (location) {
      case "/": return "Dashboard";
      case "/create-agent": return "Create Agent";
      case "/chat": return "Chat";
      case "/documents": return "Documents";
      default: return "Dashboard";
    }
  };

  return (
    <header className="bg-white border-b border-gray-200 px-6 py-4 lg:pl-72">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          {/* Mobile menu button only */}
          <div className="lg:hidden">
            <Sheet open={open} onOpenChange={setOpen}>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon">
                  <Menu className="h-6 w-6" />
                </Button>
              </SheetTrigger>
              <SheetContent side="left" className="w-64 p-0">
                <div className="bg-white h-full">
                  <div className="p-6">
                    <div className="flex items-center space-x-3 mb-8">
                      <div className="w-10 h-10 gradient-blue-teal rounded-lg flex items-center justify-center">
                        <Bot className="text-white text-lg" />
                      </div>
                      <h1 className="text-xl font-semibold text-slate-800">AI Agents</h1>
                    </div>
                    
                    <nav className="space-y-2">
                      {navigation.map((item) => {
                        const isActive = location === item.href;
                        const Icon = item.icon;
                        
                        return (
                          <Link key={item.name} href={item.href}>
                            <div
                              className={`nav-link ${isActive ? 'active' : ''}`}
                              onClick={() => setOpen(false)}
                            >
                              <Icon className="w-5 h-5" />
                              <span>{item.name}</span>
                            </div>
                          </Link>
                        );
                      })}
                    </nav>
                  </div>
                </div>
              </SheetContent>
            </Sheet>
          </div>
          <h2 className="text-lg font-semibold text-slate-800">{getPageTitle()}</h2>
        </div>
        <div className="flex items-center space-x-4">
          <Button variant="ghost" size="icon" className="relative">
            <Bell className="h-5 w-5 text-slate-600" />
            <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full"></span>
          </Button>
          <div className="w-8 h-8 gradient-blue-teal rounded-full flex items-center justify-center">
            <User className="text-white text-sm" />
          </div>
        </div>
      </div>
    </header>
  );
}
