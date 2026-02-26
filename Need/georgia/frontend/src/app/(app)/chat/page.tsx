// src/app/(app)/chat/page.tsx - New Chat Interface with Streaming
"use client";

import withAuth from "@/components/auth/withAuth";
import React, { useState, useEffect, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Send, User, Loader2, MessageCircle, MessageSquare } from 'lucide-react'; // Import MessageCircle
import { cn } from "@/lib/utils";
import { useRouter } from 'next/navigation';
import ReactMarkdown, { Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { authenticatedFetch } from "@/lib/axiosInstance";
import { useAuth } from "@/context/AuthContext";

interface Message {
  sender: 'user' | 'ai';
  text: string;
  timestamp?: string;
  references?: never;
  responseTime?: string;
}

function NewChatPage() {
  const queryClient = useQueryClient();
  const router = useRouter();
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isAiGenerating, setIsAiGenerating] = useState(false);
  const [currentStatus, setCurrentStatus] = useState('');
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const { user } = useAuth();

  const markdownComponents: Components = {};

  const getUserInitials = (firstName?: string, lastName?: string) => {
    const first = firstName?.charAt(0) || '';
    const last = lastName?.charAt(0) || '';
    return (first + last).toUpperCase() || 'U';
  };

  const handleSendMessage = async (e?: React.FormEvent<HTMLFormElement>) => {
    e?.preventDefault();
    const query = inputValue.trim();
    if (!query || isAiGenerating) return;

    const userMessage: Message = { sender: 'user', text: query };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsAiGenerating(true);
    setCurrentStatus('Initializing...');

    const aiMessagePlaceholder: Message = { sender: 'ai', text: '' };
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
      if (!response.body) throw new Error('Response body is null');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let newSessionId: string | null = null;
      let currentAiText = '';

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
                setMessages(prev => prev.map((msg, index) =>
                  index === prev.length - 1 ? { ...msg, text: currentAiText } : msg
                ));
              } else if (jsonData.event === 'done' && jsonData.session_id) {
                newSessionId = jsonData.session_id;
              } else if (jsonData.error) {
                console.error("Streaming error from backend:", jsonData.error);
                setMessages(prev => prev.map((msg, index) =>
                  index === prev.length - 1 ? { ...msg, text: `Error: ${jsonData.error}` } : msg
                ));
                done = true;
                break;
              }
            } catch (e) {
              if (!(e instanceof SyntaxError && line.substring(6).trim() === '')) {
                console.error('Error parsing SSE data:', e, 'Raw line:', line);
              }
            }
          }
        }
        if (newSessionId && done) break;
      }

      if (newSessionId) {
        queryClient.invalidateQueries({ queryKey: ['chatSessions'] });
        router.push(`/chat/${newSessionId}`);
      } else if (!messages.some(m => m.text.startsWith('Error:'))) {
        console.error("Backend did not return a session ID in the 'done' event.");
        setMessages(prev => prev.map((msg, index) =>
          index === prev.length - 1 ? { ...msg, text: `Error: Could not start new chat session.` } : msg
        ));
      }

    } catch (error: unknown) {
      console.error("Send message fetch error:", error);
      setMessages(prev => prev.map((msg, index) =>
        index === prev.length - 1 ? { ...msg, text: `Error: ${error instanceof Error ? error.message : 'Failed to get response'}` } : msg
      ));
    } finally {
      setIsAiGenerating(false);
      setCurrentStatus('');
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  return (
    <div className="flex flex-col h-full bg-[#FFFFFF] relative font-sans max-w-[1084px] mx-auto">
      {!messages.length && !isAiGenerating ? (
        /* Final Centered Hero Section + Input (Same to Same with Screenshot) */
        <div className="flex-grow flex flex-col items-center px-4 mt-36">
          <div className="flex flex-col items-center mb-10">
            {/* Primary Hero Icon in Circle */}
            <div className="w-[100px] h-[100px] bg-[#D1EAEA] rounded-full flex items-center justify-center mb-6 shadow-sm">
              <MessageCircle size={48} className="text-[#255c5d]" strokeWidth={1.5} />
            </div>

            {/* Hero Text (Georgia) */}
            <h1
              className="text-[28px] font-bold text-[#255c5d]  text-center"
              style={{ fontFamily: 'var(--unnamed-font-family-georgia)' }}
            >
              How Can I Help You Today?
            </h1>

            {/* Subtext Description (Montserrat) */}
            <p
              className="text-[14px] text-[#707070] text-center font-medium"
              style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
            >
              Ask a question about your document to get started
            </p>
          </div>

          {/* Pill-Shaped Input Field */}
          <div className="w-full max-w-[1084px] mt-2">
            <form onSubmit={handleSendMessage} className="relative w-full">
              <Input
                value={inputValue}
                onChange={handleInputChange}
                placeholder="Ask a question about your documents.."
                className="w-full h-12 pl-8 pr-16 rounded-full border-[#E0E0E0] focus:ring-[#255c5d] focus:border-[#255c5d] shadow-[0_4px_10px_rgba(0,0,0,0.05)] text-[16px] bg-white transition-all placeholder:text-[#707070] placeholder:font-medium"
                disabled={isAiGenerating}
                style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
              />
              <button
                type="submit"
                disabled={!inputValue.trim() || isAiGenerating}
                className="absolute right-3 top-1/2 -translate-y-1/2 w-[32px] h-[32px] rounded-full bg-[#D1EAEA] flex items-center justify-center text-[#255c5d] hover:bg-[#255c5d] hover:text-white transition-all active:scale-95 shadow-sm"
              >
                <Send className="w-5 h-5" />
              </button>
            </form>
          </div>
        </div>
      ) : (
        /* Standard Message Flow */
        <>
          <div className="flex-1 overflow-y-auto p-8 space-y-6">
            {messages.length === 0 && (
              <div className="flex justify-center items-center h-full">
                <Loader2 className="h-8 w-8 animate-spin text-[#255c5d]" />
              </div>
            )}

            {messages.length === 0 && (
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

            {messages.length > 0 && messages.map((msg, index) => (
              <div
                key={`msg-fallback-${index}`}
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
                    </div>
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
                disabled={isAiGenerating}
                style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
              />
              <button
                type="submit"
                disabled={!inputValue.trim() || isAiGenerating}
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
        </>
      )}
    </div>
  );
}

export default withAuth(NewChatPage);
