// src/app/(app)/chat/[session_id]/page.tsx - Dynamic Chat Interface with Streaming
"use client";

import withAuth from "@/components/auth/withAuth";
import React, { useState, useEffect, useRef } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useAuth } from "@/context/AuthContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Send, User, Loader2, MessageSquare, FileText, Info } from 'lucide-react'; // Import MessageSquare and FileText
import { cn } from "@/lib/utils";
import { useParams, useSearchParams, useRouter } from 'next/navigation';
import ReactMarkdown, { Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Download } from 'lucide-react';
import { isAxiosError } from 'axios';
import axiosInstance, { authenticatedFetch } from '@/lib/axiosInstance';
import { toast } from 'sonner';
import Link from 'next/link'; // Import Link for navigation
import { Badge } from "@/components/ui/badge"; // Import Badge
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";

// Define Reference type
interface Reference {
  chunk_id: string;
  filename: string;
  start_page: number;
  end_page: number;
  distance: number;
  doc_id: string;
  gcs_uri?: string;
}

interface Message {
  id?: string;
  sender: 'user' | 'ai';
  text: string;
  timestamp?: string;
  references?: Reference[];
  responseTime?: string;
  referenceTime?: string;
  textResponseTime?: string;
  metadata?: {
    analysis?: {
      justification?: string;
      confidence_level?: string;
    }
  };
}

const getUserInitials = (firstName?: string, lastName?: string) => {
  const first = firstName?.charAt(0) || '';
  const last = lastName?.charAt(0) || '';
  return (first + last).toUpperCase() || 'U';
};

const fetchMessages = async (sessionId: string | null): Promise<Message[]> => {
  if (!sessionId) return [];
  try {

    const response = await axiosInstance.get(
      `${process.env.NEXT_PUBLIC_BACKEND_API_URL}/backend/api/v1/docs/chat/sessions/${sessionId}/messages`,
    );

    const fetchedData = response.data;
    // Ensure each message has an 'id' field, even if it's just the index as a fallback for older data
    return fetchedData.map((msg: any, index: number) => ({
      ...msg,
      id: msg.id || `msg-${index}-${new Date().getTime()}` // Fallback ID
    }));
  } catch (error) {
    console.error("Error fetching messages:", error);
    return [];
  }
};

function SessionChatPage() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const params = useParams();
  const router = useRouter(); // Add useRouter
  const searchParams = useSearchParams();
  const sessionId = params.session_id as string;
  const isNewChat = searchParams.get('chat') === 'new';

  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isAiGenerating, setIsAiGenerating] = useState(false);
  const [isLoadingReferences, setIsLoadingReferences] = useState(false);
  const [currentStatus, setCurrentStatus] = useState('');
  const [downloadingDocId, setDownloadingDocId] = useState<string | null>(null);
  const [hasInteracted, setHasInteracted] = useState(false); // Track if user has sent a message
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const { data: fetchedMessages, isLoading: isLoadingMessages, error: messagesError } = useQuery<Message[], Error>({
    queryKey: ['chatMessages', sessionId],
    queryFn: () => fetchMessages(sessionId),
    enabled: !!sessionId && !hasInteracted, // Disable query after first interaction
    staleTime: 5 * 60 * 1000,
    refetchOnWindowFocus: false,
  });

  useEffect(() => {
    if (fetchedMessages) {
      setMessages(fetchedMessages);
      // If we fetched messages and there are some, mark as interacted to prevent re-fetching
      if (fetchedMessages.length > 0) {
        setHasInteracted(true);
      }
    } else if (!isLoadingMessages) {
      setMessages([]);
    }
  }, [fetchedMessages, isLoadingMessages]);

  useEffect(() => {
    if (messagesError) console.error("Error loading messages:", messagesError);
  }, [messagesError]);

  const handleSendMessage = async (e?: React.FormEvent<HTMLFormElement>) => {
    e?.preventDefault();
    const query = inputValue.trim();
    if (!query || isAiGenerating || !sessionId) return;

    const userMessageId = `user-${new Date().getTime()}`;
    const userMessage: Message = { id: userMessageId, sender: 'user', text: query };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsAiGenerating(true);
    setIsLoadingReferences(true);
    setCurrentStatus('Initializing...');

    const aiMessagePlaceholder: Message = { id: `ai-placeholder-${new Date().getTime()}`, sender: 'ai', text: '' };
    setMessages(prev => [...prev, aiMessagePlaceholder]);

    try {
      const response = await authenticatedFetch(`${process.env.NEXT_PUBLIC_BACKEND_API_URL}/backend/api/v2/docs/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify({ query, session_id: sessionId })
      });

      if (!response.ok) {
        let errorMsg = `HTTP error! status: ${response.status}`;
        try { const errorData = await response.json(); errorMsg = errorData.msg || errorMsg; } catch { }
        throw new Error(errorMsg);
      }
      if (!response.body) throw new Error('Response body is null');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let currentAiText = '';
      let finalSessionId = sessionId;
      let finalMessageId: string | undefined = undefined;
      let textGenerationComplete = false;

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        const chunk = decoder.decode(value, { stream: !done });
        const lines = chunk.split('\n\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const jsonData = JSON.parse(line.substring(6));

              if (jsonData.status) {
                setCurrentStatus(jsonData.status);
              } else if (jsonData.chunk) {
                currentAiText += jsonData.chunk;
                setMessages(prev => {
                  if (prev.length === 0) return prev;
                  const updatedLastMessage = { ...prev[prev.length - 1], text: currentAiText };
                  return [...prev.slice(0, -1), updatedLastMessage];
                });
              } else if (jsonData.event === 'done') {
                finalSessionId = jsonData.session_id || finalSessionId;
                finalMessageId = jsonData.message_id;
                const receivedReferences = (jsonData.references && Array.isArray(jsonData.references)) ? jsonData.references as Reference[] : [];
                const responseTime = jsonData.response_time;
                const referenceTime = jsonData.reference_time;
                const textResponseTime = jsonData.text_response_time;
                const receivedMetadata = jsonData.metadata;
                const serverTextAnswer = jsonData.text_answer; // Get the authoritative text from backend

                setMessages(prev => {
                  if (prev.length === 0) return prev;
                  const lastMessage = prev[prev.length - 1];
                  if (lastMessage.sender === 'ai') {
                    // Use server text if available (it handles fallbacks correctly), otherwise rely on stream
                    let finalText = serverTextAnswer || currentAiText;

                    // Fallback logic in case server didn't send text (legacy backend)
                    if (!finalText.trim() && receivedReferences.length > 0) {
                      finalText = "I have found relevant documents but could not generate a textual summary. Please review the references below.";
                    } else if (!finalText.trim() && (!receivedReferences || receivedReferences.length === 0)) {
                      finalText = "I apologize, but I could not generate a response or find relevant documents.";
                    }

                    const updatedLastMessage = {
                      ...lastMessage,
                      id: finalMessageId || lastMessage.id,
                      text: finalText,
                      references: receivedReferences,
                      responseTime: responseTime,
                      referenceTime: referenceTime,
                      textResponseTime: textResponseTime,
                      metadata: receivedMetadata
                    };
                    return [...prev.slice(0, -1), updatedLastMessage];
                  }
                  return prev;
                });

                queryClient.invalidateQueries({ queryKey: ['chatSessions'] });
                // Don't invalidate messages query since we're managing state locally now
              } else if (jsonData.error) {
                console.error("Streaming error from backend:", jsonData.error);
                setMessages(prev => {
                  if (prev.length === 0) return prev;
                  const updatedLastMessage = { ...prev[prev.length - 1], text: `Error: ${jsonData.error}` };
                  return [...prev.slice(0, -1), updatedLastMessage];
                });
              }
            } catch (e) {
              if (!(e instanceof SyntaxError && line.substring(6).trim() === '')) {
                console.error('Error parsing SSE data:', e, 'Raw line:', line);
              }
            }
          }
        }
      }
    } catch (error: unknown) {
      let errorMsg = 'Failed to get response';
      if (error instanceof Error) {
        errorMsg = error.message;
      }
      console.error("Send message fetch error:", error);
      setMessages(prev => prev.map((msg, index) =>
        index === prev.length - 1 ? { ...msg, text: `Error: ${errorMsg}` } : msg
      ));
    } finally {
      setIsAiGenerating(false);
      setIsLoadingReferences(false);
      setCurrentStatus('');
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const handleDownloadReference = async (docId: any, start_page: any) => {
    if (!docId || downloadingDocId === docId) return;
    setDownloadingDocId(docId);
    try {
      const response = await axiosInstance.get(`/docs/download/${docId}`);
      const { signed_url } = response.data;
      if (!signed_url) throw new Error("Backend did not provide a download URL.");
      window.open(`${signed_url}#page=${start_page}`, '_blank', 'noopener,noreferrer');
    } catch (err: unknown) {
      console.error("Download error:", err);
      let errorMsg = "Failed to get download link.";
      if (isAxiosError(err) && err.response?.data?.msg) errorMsg = err.response.data.msg;
      else if (err instanceof Error) errorMsg = err.message;
      toast.error("Download Failed", { description: errorMsg });
    } finally {
      setDownloadingDocId(null);
    }
  };

  const markdownComponents: Components = {};

  const ReferencesSection: React.FC<{ references: Reference[]; responseTime?: string; referenceTime?: string; justification?: string; confidenceLevel?: string; }> = ({ references, responseTime, referenceTime, justification, confidenceLevel }) => {
    // Debugging: Log incoming references
    // console.log("ReferencesSection received:", references);

    const validReferences = references?.filter(ref => ref.filename && ref.filename !== "Filename Unavailable") || [];
    const shouldShowReferences = validReferences.length > 0;

    if (!shouldShowReferences && !responseTime) return null;

    let badgeClass = "bg-gray-100 text-gray-800 hover:bg-gray-200"; // Default
    if (confidenceLevel === 'High') badgeClass = "bg-green-100 text-green-800 hover:bg-green-200 border-green-200";
    else if (confidenceLevel === 'Medium') badgeClass = "bg-yellow-100 text-yellow-800 hover:bg-yellow-200 border-yellow-200";
    else if (confidenceLevel === 'Low') badgeClass = "bg-red-100 text-red-800 hover:bg-red-200 border-red-200";

    return (
      <div className="flex flex-col">
        {confidenceLevel !== 'High' && (
          <p className="text-[11px] text-gray-500 mb-2 italic">
            Responses may not always be accurate. Please verify important details.
          </p>
        )}
        {shouldShowReferences && (
          <>
            <p className="">References:</p>
            <div className="flex flex-col gap-1">
              {validReferences.map((ref, index) => {
                const fileNameExtra = (ref.start_page === ref.end_page ?
                  ` [ page ${ref.start_page} ]` :
                  ` [ page ${ref.start_page} - ${ref.end_page} ]`);

                const hasDocId = !!ref.doc_id;

                return (
                  <div key={ref.chunk_id || `ref-${index}`} className="text-sm flex items-center flex-wrap gap-2 font-semibold">
                    {/* {confidenceLevel && (
                      <Badge variant="outline" className={cn("text-[10px] px-1.5 py-0 h-5 font-medium border", badgeClass)}>
                        {confidenceLevel}
                      </Badge>
                    )} */}
                    <div className="flex items-center">
                      {/* {justification && (
                        <HoverCard>
                          <HoverCardTrigger asChild>
                            <Info className="h-3.5 w-3.5 text-gray-400 hover:text-gray-600 mr-2 cursor-help" />
                          </HoverCardTrigger>
                          <HoverCardContent className="w-80 text-sm bg-white p-3 shadow-md border rounded-md z-50">
                            <p className="font-semibold mb-1 text-gray-700">Justification:</p>
                            <p className="text-gray-600 leading-snug">{justification}</p>
                          </HoverCardContent>
                        </HoverCard>
                      )} */}
                      <a
                        href="#"
                        className={cn(
                          "inline-flex items-center !text-[#317d9c]",
                          hasDocId ? "text-accent-blue hover:underline cursor-pointer" : "text-gray-600 cursor-default",
                          downloadingDocId === ref.doc_id && "opacity-50 cursor-not-allowed"
                        )}
                        onClick={(e) => {
                          e.preventDefault();
                          if (hasDocId && downloadingDocId !== ref.doc_id) {
                            handleDownloadReference(ref.doc_id, ref.start_page);
                          }
                        }}
                        title={hasDocId ? `Download ${ref.filename}${fileNameExtra}` : "Download unavailable"}
                      >
                        {/* {downloadingDocId === ref.doc_id ? (
                          <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                        ) : (
                          <Download className={cn("h-3 w-3 mr-1", !hasDocId && "text-gray-400")} />
                        )} */}
                        {ref.filename}
                      </a>
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>
    );
  };

  // Only show loading spinner on initial load if we haven't interacted yet and it's not a new chat
  const shouldShowInitialLoading = isLoadingMessages && !isNewChat && !hasInteracted && messages.length === 0;

  return (
    <div className="flex flex-col h-full bg-white relative max-w-[1084px] mx-auto">
      {/* Message Display Area */}
      <div className="flex-1 overflow-y-auto p-8 space-y-6">
        {isLoadingMessages && !isNewChat && messages.length === 0 && (
          <div className="flex justify-center items-center h-full">
            <Loader2 className="h-8 w-8 animate-spin text-[#255c5d]" />
          </div>
        )}

        {!isLoadingMessages && messages.length === 0 && !isNewChat && (
          <div className="flex flex-grow flex-col items-center justify-center -mt-20 h-full">
            <div className="w-20 h-20 bg-[#EEF4F4] rounded-full flex items-center justify-center mb-8">
              <MessageSquare size={36} className="text-[#255c5d]" />
            </div>
            <h1 className="text-[32px] font-bold text-[#255c5d] mb-4 text-center" style={{ fontFamily: 'var(--unnamed-font-family-georgia)' }}>
              How Can I Help You Today?
            </h1>
            <p className="text-[13px] text-[#707070] font-medium" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>
              Ask a question about your document to get started
            </p>
          </div>
        )}

        {!shouldShowInitialLoading && messages.length > 0 && messages.map((msg, index) => (
          <div
            key={msg.id || `msg-fallback-${index}`}
            className={cn("flex items-start gap-4", msg.sender === 'user' ? "justify-end" : "justify-start")}
          >
            {msg.sender === 'ai' && (
              <>
                <div className="w-8 h-8 bg-[#EEF4F4] rounded-full flex items-center justify-center flex-shrink-0 shadow-sm">
                  <MessageSquare size={16} className="text-[#255c5d]" />
                </div>
              </>
            )}

            <div
              className={cn(
                "py-3 px-6 rounded-[10px] max-w-[80%] relative group bg-[#F6F6F6]",
              )}
            >
              {index === messages.length - 1 && isAiGenerating && !msg.text ? (
                <div className="flex items-center gap-2 text-sm opacity-70 italic">
                  <Loader2 size={14} className="animate-spin" />
                  {currentStatus || 'Generating...'}
                </div>
              ) : (
                <div>
                  <div className="prose prose-sm max-w-none prose-p:leading-relaxed">
                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                      {`${msg.text}${index === messages.length - 1 && isAiGenerating ? ' ' : ''}`}
                    </ReactMarkdown>
                    {index === messages.length - 1 && isAiGenerating && (
                      <span className="inline-block w-2 h-2 bg-current rounded-full animate-pulse ml-1" />
                    )}
                  </div>
                  <ReferencesSection
                    references={msg.references || []}
                    responseTime={msg.responseTime}
                    referenceTime={msg.referenceTime}
                    justification={msg.metadata?.analysis?.justification}
                    confidenceLevel={msg.metadata?.analysis?.confidence_level}
                  />
                </div>
              )}
              {msg.id && sessionId && (
                <Link
                  href={`/chat/${sessionId}/message/${msg.id}/logs`}
                  className="absolute -top-3 -right-3 opacity-0 group-hover:opacity-100 transition-opacity p-1.5 bg-white border border-gray-200 text-[#707070] hover:text-black rounded-full shadow-sm"
                  title="View System Logs"
                >
                  <FileText className="h-3 w-3" />
                </Link>
              )}
            </div>
            {msg.sender === 'user' && (
              <div className="w-8 h-8 rounded-full bg-[#CCE9E8] flex items-center justify-center flex-shrink-0 text-[#255C5D] text-xs font-bold">
                {getUserInitials(user?.first_name, user?.last_name)}
              </div>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area (Rounded bar like screenshot) */}
      <div className="p-8 flex w-full z-10">
        <form onSubmit={handleSendMessage} className="relative w-full">
          <Input
            value={inputValue}
            onChange={handleInputChange}
            placeholder="Ask a question about your documents..."
            className="w-full h-12 pl-8 pr-16 rounded-full border-[#E0E0E0] focus:ring-[#255c5d] focus:border-[#255c5d] shadow-[0_4px_10px_rgba(0,0,0,0.05)] text-[16px] bg-white transition-all placeholder:text-[#707070] placeholder:font-medium"
            disabled={isAiGenerating || !sessionId}
            style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
          />
          <button
            type="submit"
            disabled={!inputValue.trim() || isAiGenerating || !sessionId}
            className="absolute right-3 top-1/2 -translate-y-1/2 w-[32px] h-[32px] rounded-full bg-[#D1EAEA] flex items-center justify-center text-[#255c5d] hover:bg-[#255c5d] hover:text-white transition-all active:scale-95 shadow-sm"
          >
            {isAiGenerating ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Send className="h-5 w-5" />
            )}
          </button>
        </form>
      </div>
    </div>
  );
}

export default withAuth(SessionChatPage);