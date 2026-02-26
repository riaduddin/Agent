import React, { useState } from "react";
import Link from "next/link";
import { useRouter, useParams, usePathname } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { isAxiosError } from "axios";
import axiosInstance from "@/lib/axiosInstance";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Loader2,
  MessageSquareMore,
  Upload,
  History,
  X,
  MoreHorizontal,
  Edit,
  Trash2,
  FileText,
  SquarePen,
  ArrowLeft,
  FilesIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { useFilePermissions } from "@/hooks/use-file-permissions";
import { Input } from "@/components/ui/input";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface ChatSession {
  id: string;
  title: string;
  last_updated: string | null;
}

interface SidebarProps {
  isOpen: boolean;
  toggleSidebar: () => void;
}

const fetchChatSessions = async (): Promise<ChatSession[]> => {
  const response = await axiosInstance.get("/docs/chat/sessions");
  return response.data;
};

const renameSession = async ({
  sessionId,
  newTitle,
}: {
  sessionId: string;
  newTitle: string;
}) => {
  const response = await axiosInstance.patch(
    `/docs/chat/sessions/${sessionId}`,
    { title: newTitle }
  );
  return response.data;
};

const deleteSession = async (sessionId: string) => {
  const response = await axiosInstance.delete(
    `/docs/chat/sessions/${sessionId}`
  );
  return response.data;
};

const Sidebar: React.FC<SidebarProps> = ({ isOpen, toggleSidebar }) => {
  const router = useRouter();
  const params = useParams();
  const pathname = usePathname();
  const queryClient = useQueryClient();
  const activeSessionId = params.session_id as string | undefined;
  const { canUpload } = useFilePermissions();

  const [renamingSessionId, setRenamingSessionId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");

  const {
    data: sessions,
    isLoading,
  } = useQuery<ChatSession[], Error>({
    queryKey: ["chatSessions"],
    queryFn: fetchChatSessions,
    staleTime: 1 * 60 * 1000,
  });

  const renameMutation = useMutation({
    mutationFn: renameSession,
    onMutate: async ({ sessionId, newTitle }) => {
      await queryClient.cancelQueries({ queryKey: ["chatSessions"] });
      const previousSessions = queryClient.getQueryData<ChatSession[]>(["chatSessions"]);
      if (previousSessions) {
        queryClient.setQueryData(
          ["chatSessions"],
          previousSessions.map((s) => (s.id === sessionId ? { ...s, title: newTitle } : s))
        );
      }
      setRenamingSessionId(null);
      return { previousSessions };
    },
    onError: (err: unknown, variables, context) => {
      let errorMsg = "Failed to rename session.";
      if (isAxiosError(err) && err.response?.data?.msg) errorMsg = err.response.data.msg;
      toast.error("Rename Failed", { description: errorMsg });
      if (context?.previousSessions) queryClient.setQueryData(["chatSessions"], context.previousSessions);
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["chatSessions"] }),
  });

  const deleteMutation = useMutation({
    mutationFn: deleteSession,
    onMutate: async (id) => {
      await queryClient.cancelQueries({ queryKey: ["chatSessions"] });
      const previousSessions = queryClient.getQueryData<ChatSession[]>(["chatSessions"]);
      if (previousSessions) {
        queryClient.setQueryData(
          ["chatSessions"],
          previousSessions.filter((s) => s.id !== id)
        );
      }
      if (activeSessionId === id) router.push("/chat");
      return { previousSessions };
    },
    onError: (err: unknown, id, context) => {
      if (context?.previousSessions) queryClient.setQueryData(["chatSessions"], context.previousSessions);
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["chatSessions"] }),
  });

  const handleNewChat = () => {
    router.push("/chat");
    if (isOpen) toggleSidebar();
  };

  const handleSessionClick = (id: string) => {
    if (renamingSessionId === id) return;
    router.push(`/chat/${id}`);
    if (isOpen) toggleSidebar();
  };

  const startRename = (s: ChatSession) => {
    setRenamingSessionId(s.id);
    setRenameValue(s.title);
  };

  return (
    <aside
      className={cn(
        "h-full bg-[#F6F6F6] border-r border-[#E0E0E0] p-0 flex flex-col z-40 transition-all duration-300 ease-in-out md:relative absolute",
        isOpen ? "w-[240px] translate-x-0" : "w-[68px] -translate-x-full md:translate-x-0"
      )}
    >
      <button
        onClick={toggleSidebar}
        className="absolute top-4 right-4 text-[#707070] md:hidden focus:outline-none"
      >
        <X className="w-5 h-5" />
      </button>

      {/* NEW CHAT SECTION */}
      <TooltipProvider>
        <div className={cn("pt-[25px] mb-[24px] flex justify-center", isOpen ? "px-[26px]" : "px-0")}>
          <Tooltip delayDuration={100}>
            <TooltipTrigger asChild>
              <button
                onClick={handleNewChat}
                className={cn("flex items-center group focus:outline-none", isOpen ? "w-full" : "justify-center")}
              >
                <div className="w-[50px] h-[46px] bg-[#255c5d] rounded-[10px] flex items-center justify-center transition-transform active:scale-95 shadow-sm shrink-0">
                  <SquarePen className="w-[24px] h-[24px] text-white" />
                </div>
                {isOpen && (
                  <span
                    className="ml-[10px] text-[13px] font-bold text-[#000000] uppercase tracking-[0.02em] whitespace-nowrap"
                    style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
                  >
                    NEW CHAT
                  </span>
                )}
              </button>
            </TooltipTrigger>
            {!isOpen && (
              <TooltipContent side="right" className="font-bold text-[11px] p-2 bg-[#255c5d] text-white border-none">
                New chat
              </TooltipContent>
            )}
          </Tooltip>
        </div>
      </TooltipProvider>

      {/* CHAT HISTORY HEADER */}
      <div className={cn("flex items-center gap-[8px] mb-[12px]", isOpen ? "px-[26px]" : "justify-center")}>
        <MessageSquareMore className="w-[16px] h-[16px] text-[#2D2D2D]" strokeWidth={3} />
        {isOpen && (
          <span
            className="text-[12px] font-bold text-[#2D2D2D] whitespace-nowrap"
            style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
          >
            Chat History
          </span>
        )}
      </div>

      {/* CHAT HISTORY LIST */}
      <nav className="chat-history-scrollbar flex-grow overflow-y-auto space-y-[1px] mb-[20px]">
        {isLoading && (
          <div className="flex justify-center py-4">
            <Loader2 className="w-4 h-4 animate-spin text-[#255c5d]/40" />
          </div>
        )}
        {!isLoading && sessions?.map((s) => {
          const isActive = activeSessionId === s.id;
          const isRenaming = renamingSessionId === s.id;

          return (
            <div key={s.id} className="group relative flex items-center h-[34px] transition-all justify-center">
              {isOpen && isActive && <div className="absolute left-0 top-0 bottom-0 w-[3px] bg-[#005D5E] z-10" />}

              {isOpen && (isRenaming ? (
                <div className="flex-1 px-4 z-20">
                  <Input
                    className="h-7 text-[13px] py-1 px-2 focus-visible:ring-1 focus-visible:ring-[#255c5d]"
                    value={renameValue}
                    onChange={(e) => setRenameValue(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") renameMutation.mutate({ sessionId: s.id, newTitle: renameValue });
                      if (e.key === "Escape") setRenamingSessionId(null);
                    }}
                    onBlur={() => setRenamingSessionId(null)}
                    autoFocus
                  />
                </div>
              ) : (
                <button
                  onClick={() => handleSessionClick(s.id)}
                  className={cn(
                    "flex-1 text-left py-[7px] !text-[14px] truncate transition-colors flex items-center",
                    isOpen ? "pl-[26px] pr-[40px]" : "justify-center",
                    isActive
                      ? "bg-[#C6E3E2] text-[#255c5d] font-bold"
                      : "text-[#808080] hover:text-[#000000] hover:bg-white/50"
                  )}
                  style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
                >
                  {isOpen ? <span className="block max-w-[150px] truncate">{s.title}</span> : null}
                </button>
              ))}

              {isOpen && !isRenaming && (
                <div className="absolute right-[12px]">
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        variant="ghost"
                        className={cn(
                          "h-6 w-6 p-0 border-none focus-visible:ring-0 focus:bg-transparent",
                          isActive ? "opacity-100 text-[#255c5d]" : "opacity-0 group-hover:opacity-100 text-[#808080]"
                        )}
                      >
                        <MoreHorizontal className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-[120px]">
                      <DropdownMenuItem onClick={() => startRename(s)} className="text-[11px] font-medium">
                        <Edit className="mr-1 h-3.5 w-3.5 text-[#000000]" /> Rename
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => deleteMutation.mutate(s.id)} className="text-[11px] font-medium">
                        <Trash2 className="mr-1 h-3.5 w-3.5 text-[#000000]" /> Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              )}
            </div>
          );
        })}
      </nav>

      {/* BOTTOM NAVIGATION */}
      <TooltipProvider>
        <div className="mt-auto flex flex-col pt-[10px]">
          {[
            { href: "/file-management", label: "FILE MANAGEMENT", tooltip: "File management", icon: FileText, show: true },
            { href: "/transfer-history", label: "TRANSFER HISTORY", tooltip: "Transfer history", icon: FilesIcon, show: true },
            { href: "/upload", label: "UPLOAD FILE", tooltip: "File upload", icon: Upload, show: canUpload, },
            { href: "/history", label: "UPLOAD HISTORY", tooltip: "Upload history", icon: History, show: true },
          ].filter(item => item.show).map((item, idx) => {
            const isActive = pathname === item.href;
            return (
              <Tooltip key={item.href} delayDuration={100}>
                <TooltipTrigger asChild>
                  <Link
                    href={item.href}
                    className={cn(
                      "relative flex items-center h-[40px] transition-all group border-b border-[#dedede] overflow-hidden",
                      isOpen ? "pl-[26px]" : "justify-center",
                      idx === 0 && "border-t border-[#dedede]"
                    )}
                  >
                    {isActive && <div className="absolute left-0 top-0 bottom-0 w-[4px] bg-[#005F85]" />}
                    <item.icon className="w-[18px] h-[18px] text-[#005F85] shrink-0" />
                    {isOpen && (
                      <span
                        className="ml-[12px] text-[11px] font-bold text-[#005F85] tracking-[0.05em] whitespace-nowrap"
                        style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
                      >
                        {item.label}
                      </span>
                    )}
                  </Link>
                </TooltipTrigger>
                {!isOpen && (
                  <TooltipContent side="right" className="font-bold text-[11px] p-2 bg-[#255c5d] text-white border-none">
                    {item.tooltip}
                  </TooltipContent>
                )}
              </Tooltip>
            );
          })}
        </div>
      </TooltipProvider>

      {/* SIDEBAR TOGGLE BUTTON */}
      <div className="flex justify-center items-center h-[40px] border-t border-[#dedede] bg-[#F6F6F6]">
        <button
          onClick={toggleSidebar}
          className="w-[24px] h-[24px] bg-[#C6E3E2] rounded-full flex items-center justify-center transition-all hover:scale-110 active:scale-95 shadow-sm"
        >
          {isOpen ? (
            <ArrowLeft className="w-[14px] h-[14px] text-[#255c5d]" />
          ) : (
            <ArrowLeft className="w-[14px] h-[14px] text-[#255c5d] rotate-180" />
          )}
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;
