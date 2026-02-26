// src/app/(app)/page.tsx - Main Chat Interface (Handles initial chat, now streaming)
"use client";

import withAuth from "@/components/auth/withAuth";
import React, { useState, useEffect, useRef } from 'react'; // Removed useCallback
// Removed unused useQuery
import { useQueryClient } from '@tanstack/react-query';
import { useRouter } from 'next/navigation'; // Import useRouter
import { isAxiosError } from 'axios'; // Import axios and isAxiosError for download error handling
import axiosInstance, { authenticatedFetch } from '@/lib/axiosInstance'; // Need axios for download call
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Send, User, Bot, Loader2, Download, MessageSquare, Info } from 'lucide-react'; // Added Download, MessageSquare, Info
import { cn } from "@/lib/utils"; // For conditional classes
import ReactMarkdown, { Components } from 'react-markdown'; // Import ReactMarkdown
import remarkGfm from 'remark-gfm'; // Import remarkGfm
import { toast } from 'sonner'; // For showing download errors
import { Badge } from "@/components/ui/badge"; // Import Badge
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";

// Define Reference type (same as session page)
interface Reference {
  chunk_id: string;
  filename: string;
  start_page: number;
  end_page: number;
  distance: number;
  doc_id: string; // Original document ID
  gcs_uri?: string; // Original document GCS URI (optional for display)
}

interface Message {
  sender: 'user' | 'ai';
  text: string;
  timestamp?: string; // Optional for display
  references?: Reference[]; // Updated to hold Reference objects
  metadata?: {
    analysis?: {
      justification?: string;
      confidence_level?: string;
    }
  };
}

// Custom component mapping to override paragraph margins and line height (same as session page)
const markdownComponents: Components = {
  p: (props) => <p className="mb-0 leading-snug" {...props} />,
};

function ChatPage() {
  const queryClient = useQueryClient();
  const router = useRouter(); // Initialize useRouter
  const [messages, setMessages] = useState<Message[]>([]); // Local messages for this initial interaction
  const [inputValue, setInputValue] = useState('');
  const [isAiGenerating, setIsAiGenerating] = useState(false); // Loading state for AI response
  const [downloadingDocId, setDownloadingDocId] = useState<string | null>(null); // Track download state
  const messagesEndRef = useRef<HTMLDivElement | null>(null); // Ref for scrolling

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  // Rewritten handleSendMessage using fetch for streaming
  const handleSendMessage = async (e?: React.FormEvent<HTMLFormElement>) => {
    e?.preventDefault();
    const query = inputValue.trim();
    if (!query || isAiGenerating) return; // No activeSessionId check needed here

    const userMessage: Message = { sender: 'user', text: query };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsAiGenerating(true);

    // Add placeholder for AI response
    const aiMessagePlaceholder: Message = { sender: 'ai', text: '' }; // References will be added later
    setMessages(prev => [...prev, aiMessagePlaceholder]);

    try {
      const response = await authenticatedFetch(`${process.env.NEXT_PUBLIC_BACKEND_API_URL}/backend/api/v2/docs/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify({ query, session_id: null }),
      });

      if (!response.ok) {
        let errorMsg = `HTTP error! status: ${response.status}`;
        try { const errorData = await response.json(); errorMsg = errorData.msg || errorMsg; } catch { }
        throw new Error(errorMsg);
      }

      if (!response.body) {
        throw new Error('Response body is null');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let currentAiText = '';
      let newSessionId: string | null = null;

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        const chunk = decoder.decode(value, { stream: !done });
        const lines = chunk.split('\n\n'); // SSE lines are separated by double newlines

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              // Use type assertion or validation if structure is known
              const jsonData: Record<string, unknown> = JSON.parse(line.substring(6)); // Replaced any with unknown

              // Directly update references on the last message when the event arrives
              if (jsonData.references && Array.isArray(jsonData.references)) {
                const receivedReferences = jsonData.references as Reference[]; // Assert type
                setMessages(prev => {
                  if (prev.length === 0) return prev; // Should not happen
                  const updatedLastMessage = { ...prev[prev.length - 1], references: receivedReferences };
                  return [...prev.slice(0, -1), updatedLastMessage]; // Create new array
                });
              } else if (jsonData.chunk) {
                // Append text chunk only if it's a string
                if (typeof jsonData.chunk === 'string') {
                  currentAiText += jsonData.chunk;
                  setMessages(prev => {
                    if (prev.length === 0) return prev; // Should not happen
                    const updatedLastMessage = { ...prev[prev.length - 1], text: currentAiText };
                    return [...prev.slice(0, -1), updatedLastMessage]; // Create new array
                  });
                }
              } else if (jsonData.event === 'done' && jsonData.session_id) {
                newSessionId = jsonData.session_id as string; // Added type assertion

                // Capture metadata if present
                const receivedMetadata = jsonData.metadata as Message['metadata'];
                if (receivedMetadata) {
                  setMessages(prev => {
                    if (prev.length === 0) return prev;
                    const updatedLastMessage = { ...prev[prev.length - 1], metadata: receivedMetadata };
                    return [...prev.slice(0, -1), updatedLastMessage];
                  });
                }
              } else if (jsonData.error) {
                console.error("Streaming error from backend:", jsonData.error);
                // Use the slice/append pattern for error update too for consistency
                setMessages(prev => {
                  if (prev.length === 0) return prev;
                  const updatedLastMessage = { ...prev[prev.length - 1], text: `Error: ${jsonData.error}` };
                  return [...prev.slice(0, -1), updatedLastMessage];
                });
              }
            } catch (e: unknown) { // Use unknown type for error
              if (!(e instanceof SyntaxError && line.substring(6).trim() === '')) {
                console.error('Error parsing SSE data:', e, 'Raw line:', line);
              }
            }
          }
        }
      }

      // After stream is finished, if we got a new session ID, navigate
      if (newSessionId) {
        queryClient.invalidateQueries({ queryKey: ['chatSessions'] }); // Refresh sidebar
        router.push(`/chat/${newSessionId}?chat=new`); // Navigate with query param
      } else {
        console.error("Stream finished but no new session ID received.");
        setMessages(prev => prev.map((msg, index) =>
          index === prev.length - 1 ? { ...msg, text: "Error: Could not establish new chat session." } : msg
        ));
      }

    } catch (error: unknown) { // Use unknown type
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
    }
  };

  // Download handler (same as session page)
  const handleDownloadReference = async (docId: string, filename: string) => {
    if (!docId || downloadingDocId === docId) return; // Prevent double clicks
    setDownloadingDocId(docId);
    try {
      const response = await axiosInstance.get(`/docs/download/${docId}`);
      const { signed_url, filename: downloadFilename } = response.data;

      if (!signed_url) {
        throw new Error("Backend did not provide a download URL.");
      }

      // Open in new tab instead of forcing download
      window.open(signed_url, '_blank', 'noopener,noreferrer');

    } catch (err: unknown) { // Use unknown type
      console.error("Download error:", err);
      let errorMsg = "Failed to get download link.";
      // Use type guard for AxiosError
      if (isAxiosError(err) && err.response?.data?.msg) {
        errorMsg = err.response.data.msg;
      } else if (err instanceof Error) {
        errorMsg = err.message;
      }
      toast.error("Download Failed", { description: errorMsg });
    } finally {
      setDownloadingDocId(null);
    }
  };


  return (
    <div className="flex flex-col h-full bg-white relative">
      {/* Message Display Area */}
      <div className="flex-1 overflow-y-auto p-8 space-y-6">
        {/* Show placeholder if no messages and not generating */}
        {messages.length === 0 && !isAiGenerating && (
          <div className="flex-grow flex flex-col items-center justify-center -mt-20 h-full">
            <div className="w-24 h-24 bg-[#D1EAEA] rounded-full flex items-center justify-center mb-8">
              <MessageSquare size={44} className="text-[#255c5d]" />
            </div>

            <h1
              className="text-[34px] font-bold text-[#255c5d] mb-3"
              style={{ fontFamily: 'var(--unnamed-font-family-georgia)' }}
            >
              How Can I Help You Today?
            </h1>

            <p
              className="text-sm text-[#000000] opacity-80"
              style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
            >
              Ask a question about your documents to get started
            </p>
          </div>
        )}

        {/* Display messages stored in local state for this initial interaction */}
        {messages.length > 0 && messages.map((msg, index) => (
          <div
            key={index}
            className={cn(
              "flex items-start gap-4",
              msg.sender === 'user' ? "justify-end" : "justify-start"
            )}
          >
            {msg.sender === 'ai' && (
              <div className="w-8 h-8 rounded-full bg-[#255c5d] text-white flex items-center justify-center flex-shrink-0 text-xs font-bold shadow-sm">
                <span className="translate-y-[1px]">GA</span>
              </div>
            )}
            <div
              className={cn(
                "p-4 rounded-2xl max-w-[80%] shadow-sm relative group",
                msg.sender === 'user'
                  ? "bg-[#255c5d] text-white rounded-tr-none"
                  : "bg-[#F4F3F0] text-black rounded-tl-none border border-gray-100"
              )}
            >
              {/* Render AI message using Markdown */}
              {msg.sender === 'ai' ? (
                index === messages.length - 1 && isAiGenerating && !msg.text ? (
                  <div className="flex items-center gap-2 text-sm opacity-70 italic">
                    <Loader2 size={14} className="animate-spin" />
                    Generating...
                  </div>
                ) : (
                  // Container for Markdown content and References
                  <div>
                    {/* Markdown Content */}
                    <div className="prose prose-sm max-w-none prose-p:leading-relaxed">
                      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                        {`${msg.text}${index === messages.length - 1 && isAiGenerating ? ' ' : ''}`}
                      </ReactMarkdown>
                      {index === messages.length - 1 && isAiGenerating && (
                        <span className="inline-block w-2 h-2 bg-current rounded-full animate-pulse ml-1" />
                      )}
                    </div>
                    {/* References Section */}
                    {msg.references && msg.references.length > 0 &&
                      !msg.text.startsWith("I cannot answer") &&
                      !msg.text.startsWith("I could not find") &&
                      (
                        <div className="mt-3 pt-3 border-t border-gray-200">
                          <p className="text-xs font-semibold mb-2 text-gray-500 uppercase tracking-wide">References</p>
                          <div className="flex flex-col gap-1.5">
                            {msg.references.map((ref, refIndex) => {
                              const confidence = msg.metadata?.analysis?.confidence_level;
                              let badgeClass = "bg-gray-100 text-gray-800 border-gray-200";
                              if (confidence === 'High') badgeClass = "bg-green-50 text-green-700 border-green-200";
                              else if (confidence === 'Medium') badgeClass = "bg-yellow-50 text-yellow-700 border-yellow-200";
                              else if (confidence === 'Low') badgeClass = "bg-red-50 text-red-700 border-red-200";

                              return (
                                <div key={ref.chunk_id || refIndex} className="text-xs flex items-center flex-wrap gap-2">
                                  {confidence && (
                                    <Badge variant="outline" className={cn("text-[10px] px-1.5 py-0 h-5 font-medium border", badgeClass)}>
                                      {confidence}
                                    </Badge>
                                  )}
                                  <div className="flex items-center">
                                    {msg.metadata?.analysis?.justification && (
                                      <HoverCard>
                                        <HoverCardTrigger asChild>
                                          <Info className="h-3.5 w-3.5 text-gray-400 hover:text-gray-600 mr-2 cursor-help" />
                                        </HoverCardTrigger>
                                        <HoverCardContent className="w-80 text-sm bg-white p-3 shadow-md border rounded-md z-50">
                                          <p className="font-semibold mb-1 text-gray-700">Justification:</p>
                                          <p className="text-gray-600 leading-snug">{msg.metadata?.analysis?.justification}</p>
                                        </HoverCardContent>
                                      </HoverCard>
                                    )}
                                    <a
                                      href="#"
                                      className="inline-flex items-center text-[#255c5d] hover:underline cursor-pointer font-medium"
                                      onClick={(e) => {
                                        e.preventDefault();
                                        handleDownloadReference(ref.doc_id, ref.filename);
                                      }}
                                      title={`View ${ref.filename}`}
                                    >
                                      {downloadingDocId === ref.doc_id ? (
                                        <Loader2 className="h-3 w-3 mr-1.5 animate-spin" />
                                      ) : (
                                        <Download className="h-3 w-3 mr-1.5" />
                                      )}
                                      {ref.filename} <span className="text-gray-400 font-normal ml-1">(p. {ref.start_page}{ref.end_page !== ref.start_page ? `-${ref.end_page}` : ''})</span>
                                    </a>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                  </div>
                )
              ) : (
                msg.text
              )}
            </div>
            {msg.sender === 'user' && (
              <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center flex-shrink-0 text-[#707070]">
                <User size={18} />
              </div>
            )}
          </div>
        ))}
        {/* Empty div to scroll to */}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-10 flex justify-center w-full bg-white z-10 border-t border-gray-100">
        <form onSubmit={handleSendMessage} className="relative w-full max-w-5xl">
          <Input
            value={inputValue}
            onChange={handleInputChange}
            placeholder="Ask a question about your documents..."
            className="w-full h-14 pl-6 pr-14 rounded-2xl border-[#E0E0E0] focus:ring-[#255c5d] focus:border-[#255c5d] shadow-md text-base"
            disabled={isAiGenerating}
            style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
          />
          <button
            type="submit"
            disabled={!inputValue.trim() || isAiGenerating}
            className="absolute right-4 top-1/2 -translate-y-1/2 p-2 text-[#255c5d] hover:text-[#1E4E55] disabled:text-[#A0A0A0] transition-colors"
          >
            {isAiGenerating ? (
              <Loader2 className="h-6 w-6 animate-spin" />
            ) : (
              <Send className="h-6 w-6 rotate-[-45deg] translate-x-1" />
            )}
          </button>
        </form>
      </div>
    </div>
  );
}

export default withAuth(ChatPage);
