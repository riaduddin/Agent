// src/app/(app)/history/page.tsx - Updated with Pagination and Search
"use client";

import withAuth from "@/components/auth/withAuth";
import React, { useState, useEffect, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import axiosInstance from '@/lib/axiosInstance';
import { format } from 'date-fns';
import Link from 'next/link'; // Import Link
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Checkbox } from "@/components/ui/checkbox";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command";


import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card"; // Import Card components
import { Loader2, AlertCircle, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, X, Search, File, Scan, Package, Layers, RefreshCw, List, History as HistoryIcon, Eye, Download, Upload, CloudUpload, Filter, Brain, Cpu, User, FolderOpen, FileText } from 'lucide-react';
import { useDebounce } from '@/hooks/useDebounce';
import { toast } from 'sonner';
import { useAuth } from '@/context/AuthContext';
import { cn } from "@/lib/utils";

// Define the expected structure of a history item from the backend
interface HistoryItem {
  id: string;
  original_filename: string;
  upload_timestamp: string; // ISO string format from backend
  status: string; // Single status field: 'pending', 'processing', 'completed', 'error'
  error_message?: string | null;
  gcs_uri: string;
  file_size_bytes: number;
  source?: string; // Add the source field
  download_url?: string | null; // URL for viewing/downloading the file
}

// Define the structure of the API response
interface HistoryApiResponse {
  history: HistoryItem[];
  next_cursor: string | null; // ID of the last item for the next page's start_after
  limit: number;
  search_term: string | null;
  status: string[] | null;
  total_items: number; // Added for total page count
}

interface OverallStats {
  total_chunks: number;
  gemini_chunks: number;
  docai_chunks: number;
  reprocessing_queue?: number; // Added reprocessing queue count
}

// Define the fetch function with parameters
const fetchUploadHistory = async (
  limit: number,
  searchTerm: string | null,
  status: string[] | null, // Added status filter
  startAfter: string | null
): Promise<HistoryApiResponse> => {
  const params = new URLSearchParams();
  params.append('limit', String(limit));
  if (searchTerm) {
    params.append('search', searchTerm);
  }
  if (status && status.length > 0) { // Add status to params if selected
    // Map 'Ready' back to 'Completed' for the backend
    const apiStatuses = status.map(s => s === 'Ready' ? 'Completed' : s);
    params.append('status', apiStatuses.join(','));
  }
  if (startAfter) {
    params.append('start_after', startAfter);
  }

  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_API_URL;
  const response = await axiosInstance.get(`${backendUrl}/backend/api/v2/docs/history`, { params });
  return response.data;
};

const fetchOverallStats = async (): Promise<OverallStats> => {
  const response = await axiosInstance.get(`/docs/overall-processing-stats`);
  return response.data;
};

// Helper function to format timestamp
const formatTimestamp = (isoString: string): string => {
  try {
    return format(new Date(isoString), 'M/d/yyyy,h:mm a'); // More detailed format
  } catch (error) {
    console.error("Error formatting date:", error);
    return isoString; // Fallback
  }
};

// Helper function to determine status display
const getStatusDisplay = (status: string): { text: string; dotColor: string } => {
  switch (status?.toLowerCase()) {
    case 'completed':
      return { text: 'Ready', dotColor: 'bg-[#4CAF50]' }; // Green
    case 'splitting':
    case 'splitting_in_progress':
    case 'pending_chunk_processing':
    case 'ocr_pending':
    case 'ocr_in_progress':
    case 'pending_vectorization':
    case 'vectorizing':
    case 'processing':
      return { text: 'Processing', dotColor: 'bg-[#9C27B0]' }; // Purple
    case 'queued_for_splitting':
    case 'pending':
      return { text: 'Pending', dotColor: 'bg-[#F3992F]' }; // Orange
    case 'error':
    case 'failed':
    case 'upload_failed':
    case 'ocr_failed':
    case 'embedding_failed':
    case 'vectorization_failed':
    case 'processing_error':
    case 'error_creating_chunks':
    case 'error_splitting':
    case 'incomplete':
    case 'error_worker_failure':
    case 'unknown':
      return { text: 'Failed', dotColor: 'bg-[#F44336]' }; // Red
    default:
      return { text: status || 'Unknown', dotColor: 'bg-gray-400' };
  }
};

// Helper to format file size (handles potential non-numeric input)
const formatFileSize = (bytes: number | null | undefined): string => {
  // console.log("[formatFileSize] Received:", bytes, "Type:", typeof bytes); // DEBUG LOG
  // Check for null, undefined, NaN, or negative values
  if (bytes === null || bytes === undefined || isNaN(bytes) || bytes < 0) {
    // console.log("[formatFileSize] Returning N/A for:", bytes); // DEBUG LOG
    return 'N/A'; // Return 'N/A' for invalid sizes
  }
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};


// Helper component to highlight search term in text
const HighlightedText = ({ text, highlight }: { text: string; highlight: string }) => {
  if (!highlight || !highlight.trim()) {
    return <>{text}</>;
  }

  // Escape special regex characters in the highlight term
  const escapedHighlight = highlight.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const regex = new RegExp(`(${escapedHighlight})`, 'gi');
  const parts = text.split(regex);

  return (
    <>
      {parts.map((part, index) => (
        regex.test(part) ? (
          <mark key={index} className="bg-amber-200 text-amber-900 px-0.5 rounded">
            {part}
          </mark>
        ) : (
          <span key={index}>{part}</span>
        )
      ))}
    </>
  );
};

function UploadHistoryPage() {
  const { user } = useAuth();
  const isSuperAdmin = user?.role === 'superadmin';

  // Removed unused queryClient
  // const queryClient = useQueryClient();
  const [searchTerm, setSearchTerm] = useState('');
  const searchInputRef = React.useRef<HTMLInputElement>(null);
  const [statusFilter, setStatusFilter] = useState<string[]>([]); // Multi-select status filter state
  const [tempStatusFilter, setTempStatusFilter] = useState<string[]>([]); // For popover state
  const [isFilterOpen, setIsFilterOpen] = useState(false);
  const [pageSize, setPageSize] = useState(10);
  const [currentPage, setCurrentPage] = useState(1); // Track page number for display/logic
  const [pageCursors, setPageCursors] = useState<(string | null)[]>([null]); // Store cursor for start of each page [null, cursor1, cursor2,...]

  const debouncedSearchTerm = useDebounce(searchTerm, 500); // Debounce search input

  // Calculate the cursor for the current page
  const currentCursor = useMemo(() => pageCursors[currentPage - 1], [pageCursors, currentPage]);

  const queryKey = ['uploadHistory', pageSize, debouncedSearchTerm, statusFilter, currentCursor]; // Added statusFilter to queryKey

  // Simplified useQuery options
  const { data, isLoading, error, isError, isFetching } = useQuery<HistoryApiResponse, Error, HistoryApiResponse, readonly unknown[]>({
    queryKey: queryKey,
    queryFn: () => fetchUploadHistory(pageSize, debouncedSearchTerm || null, statusFilter, currentCursor), // Pass statusFilter
    // staleTime: 5 * 60 * 1000, // Optional: 5 minutes
  });

  const { data: overallStats, isLoading: isLoadingOverall, refetch: refetchOverall } = useQuery<OverallStats>({
    queryKey: ['overallProcessingStats'],
    queryFn: fetchOverallStats,
    enabled: isSuperAdmin,
  });

  const [isReprocessing, setIsReprocessing] = useState(false);

  const confirmBulkReprocess = async () => {
    setIsReprocessing(true);
    try {
      const response = await axiosInstance.post('/docs/bulk-reprocess-legacy');
      toast.success(response.data.msg || "Bulk reprocessing started.");
      refetchOverall();
    } catch (error: any) {
      toast.error(error.response?.data?.msg || "Failed to start bulk reprocessing.");
    } finally {
      setIsReprocessing(false);
    }
  };

  // Helper to handle file download without page navigation
  const handleDownload = async (url: string, filename: string) => {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error('Network response was not ok');
      const blob = await response.blob();
      const blobUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = blobUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(blobUrl);
    } catch (error) {
      console.error('Download failed:', error);
      // Fallback: Open in new tab if blob fetch fails (e.g., CORS)
      window.open(url, '_blank', 'noopener,noreferrer');
    }
  };

  const totalPages = useMemo(() => {
    if (data?.total_items && pageSize > 0) {
      return Math.ceil(data.total_items / pageSize);
    }
    return 1;
  }, [data?.total_items, pageSize]);


  // Removed unused nextCursor variable declaration
  // const nextCursor = data?.next_cursor;

  // Reset pagination when search term, page size, or status filter changes
  useEffect(() => {
    setCurrentPage(1);
    setPageCursors([null]);
  }, [debouncedSearchTerm, pageSize, statusFilter]); // Added statusFilter to dependencies

  const handleNextPage = () => {
    // Access next_cursor directly from data
    if (data?.next_cursor) {
      // Add the new cursor only if it's not already the last one we know
      if (pageCursors.length === currentPage) {
        setPageCursors(prev => [...prev, data.next_cursor ?? null]); // Ensure null if undefined
      }
      setCurrentPage(prev => prev + 1);
    }
  };

  const handlePreviousPage = () => {
    if (currentPage > 1) {
      setCurrentPage(prev => prev - 1);
    }
  };

  const handleFirstPage = () => {
    setCurrentPage(1);
    setPageCursors([null]);
  };

  const handleLastPage = () => {
    // This is tricky with cursor-based pagination unless we have all cursors
    // For now, let's skip or implement if total_items and pageSize allow it
    // But cursor pagination usually doesn't support jump to last easily
  };

  // Determine if there are more pages
  // Use data directly for hasMoreData check
  const hasMoreData = !!data?.next_cursor && data?.history?.length === pageSize;

  return (
    <div className="flex-1 bg-[#FFFFFF] p-8 overflow-y-auto">
      <div className="max-w-7xl mx-auto space-y-8">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <h1
            className="text-[32px] font-bold text-[#255c5d]"
            style={{ fontFamily: 'var(--unnamed-font-family-georgia)' }}
          >
            Upload History
          </h1>
        </div>

        {/* Global Statistics Strip - Super Admin Only */}
        {isSuperAdmin && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div style={{ background: '#FFFFFF', boxShadow: '0px 3px 6px #D0D4DC40', border: '2px solid #EEF4F4' }} className="rounded-[4px] px-5 py-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Layers className="h-4 w-4" />
                <span className="font-semibold" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Total Chunks</span>
              </div>
              <span className="inline-flex items-center justify-center min-w-[28px] h-[24px] px-2 rounded-[4px] bg-[#57B6B2] text-xs font-bold opacity-70">{overallStats?.total_chunks ?? 0}</span>
            </div>

            <div style={{ background: '#FFFFFF', boxShadow: '0px 3px 6px #D0D4DC40', border: '2px solid #EEF4F4' }} className="rounded-[4px] px-5 py-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Brain className="h-4 w-4" />
                <span className="font-semibold" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>LLM Extractions OCR</span>
              </div>
              <span className="inline-flex items-center justify-center min-w-[28px] h-[24px] px-2 rounded-[4px] bg-[#57B6B2] text-xs font-bold opacity-70">{overallStats?.gemini_chunks ?? 0}</span>
            </div>

            <div style={{ background: '#FFFFFF', boxShadow: '0px 3px 6px #D0D4DC40', border: '2px solid #EEF4F4' }} className="rounded-[4px] px-5 py-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4" />
                <span className="font-semibold" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Legacy OCR</span>
              </div>
              <span className="inline-flex items-center justify-center min-w-[28px] h-[24px] px-2 rounded-[4px] bg-[#57B6B2] text-xs font-bold opacity-70">{overallStats?.docai_chunks ?? 0}</span>
            </div>
          </div>
        )}

        {/* Search and Filters */}
        <div className="flex items-center gap-4 mb-2">
          <div className="relative w-[400px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-[16px] w-[16px] text-[#707070]" strokeWidth={2} />
            <Input
              ref={searchInputRef}
              placeholder="Search"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 h-10 bg-white border-[#D0D4DC] rounded-[4px] text-[14px] text-[#000000] shadow-none font-medium focus-visible:ring-1 focus-visible:ring-[#255c5d]"
              style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
            />
          </div>
        </div>

        {/* Table Container */}
        <div className="bg-white border border-[#E0E0E0] rounded-[8px] overflow-hidden shadow-sm">
          <Table>
            <TableHeader className="bg-[#EEF4F4]">
              <TableRow className="border-b border-[#E0E0E0] hover:bg-transparent">
                <TableHead className="text-[12px] font-medium text-[#1E4E55] py-3 pl-6">
                  <div className="flex items-center gap-2">
                    File name <Filter className="h-[14px] w-[14px] text-[#1E4E55]" />
                  </div>
                </TableHead>
                <TableHead className="text-[12px] font-medium text-[#1E4E55] py-3">
                  <div className="flex items-center gap-2">
                    Size <Filter className="h-[14px] w-[14px] text-[#1E4E55]" />
                  </div>
                </TableHead>
                <TableHead className="text-[12px] font-medium text-[#1E4E55] py-3">
                  <div className="flex items-center gap-2">
                    Date & Time <Filter className="h-[14px] w-[14px] text-[#1E4E55]" />
                  </div>
                </TableHead>
                <TableHead className="text-[12px] font-medium text-[#1E4E55] py-3">
                  <div className="flex items-center gap-2">
                    Source <Filter className="h-[14px] w-[14px] text-[#1E4E55]" />
                  </div>
                </TableHead>
                <TableHead className="text-[12px] font-medium text-[#1E4E55] py-3">
                  <div className="flex items-center gap-2">
                    Status
                    <Popover
                      open={isFilterOpen}
                      onOpenChange={(open) => {
                        setIsFilterOpen(open);
                        if (open) setTempStatusFilter(statusFilter);
                      }}
                    >
                      <PopoverTrigger asChild>
                        <button className="focus:outline-none">
                          <Filter className={cn("h-[14px] w-[14px] transition-colors", statusFilter.length > 0 ? "text-[#1E4E55]" : "text-[#064372]")} />
                        </button>
                      </PopoverTrigger>
                      <PopoverContent className="w-[280px] p-4 rounded-[12px] shadow-lg border border-[#E0E0E0] mt-2" align="start">
                        <div className="relative mb-6">
                          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[#707070]" strokeWidth={2.5} />
                          <Input
                            placeholder="Search"
                            className="pl-9 h-10 border-[#D0D4DC] rounded-[4px] text-[14px] focus-visible:ring-1 focus-visible:ring-[#1E4E55]"
                            onChange={(e) => {
                              // Optional: Implement filtering inside popover list
                            }}
                          />
                        </div>

                        <div className="space-y-3 mb-6">
                          {['All', 'Ready', 'Processing', 'Pending', 'Failed'].map((status) => (
                            <div
                              key={status}
                              className="flex items-center gap-3 cursor-pointer hover:bg-gray-50 p-1 rounded transition-colors"
                              onClick={() => {
                                if (status === 'All') {
                                  if (tempStatusFilter.length === 4) {
                                    setTempStatusFilter([]);
                                  } else {
                                    setTempStatusFilter(['Ready', 'Processing', 'Pending', 'Failed']);
                                  }
                                } else {
                                  setTempStatusFilter(prev =>
                                    prev.includes(status)
                                      ? prev.filter(s => s !== status)
                                      : [...prev, status]
                                  );
                                }
                              }}
                            >
                              <div className={cn(
                                "w-[18px] h-[18px] border-2 rounded-[2px] flex items-center justify-center transition-colors",
                                (status === 'All' ? tempStatusFilter.length === 4 : tempStatusFilter.includes(status))
                                  ? "bg-white border-[#1E4E55]"
                                  : "bg-white border-[#D0D4DC]"
                              )}>
                                {(status === 'All' ? tempStatusFilter.length === 4 : tempStatusFilter.includes(status)) && (
                                  <div className="w-[10px] h-[10px] bg-[#1E4E55] rounded-[1px]" />
                                )}
                              </div>
                              <span className="text-[14px] font-medium text-[#000000]">{status}</span>
                            </div>
                          ))}
                        </div>

                        <div className="text-[13px] text-[#707070] font-medium mb-6">
                          {tempStatusFilter.length} items selected
                        </div>

                        <div className="flex items-center gap-3">
                          <button
                            className="flex-1 h-10 bg-[#1E4E55] text-white rounded-[6px] text-[14px] font-bold tracking-wide uppercase hover:bg-[#163a40] transition-colors"
                            onClick={() => {
                              setStatusFilter(tempStatusFilter);
                              setIsFilterOpen(false);
                            }}
                          >
                            FILTER
                          </button>
                          <button
                            className="flex-1 h-10 border border-[#1E4E55] text-[#1E4E55] rounded-[6px] text-[14px] font-bold tracking-wide uppercase hover:bg-gray-50 transition-colors"
                            onClick={() => {
                              setTempStatusFilter([]);
                              setStatusFilter([]);
                              setIsFilterOpen(false);
                            }}
                          >
                            CLEAR
                          </button>
                        </div>
                      </PopoverContent>
                    </Popover>
                  </div>
                </TableHead>
                <TableHead className="text-[12px] font-medium text-[#1E4E55] py-3 text-center">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading ? (
                <TableRow>
                  <TableCell colSpan={6} className="h-32 text-center">
                    <Loader2 className="h-8 w-8 animate-spin text-[#064372] mx-auto" />
                  </TableCell>
                </TableRow>
              ) : data?.history && data.history.length > 0 ? (
                data.history.map((item) => {
                  const displayStatus = getStatusDisplay(item.status);

                  return (
                    <TableRow key={item.id} className="border-b border-[#E0E0E0] hover:bg-gray-50 transition-colors h-[48px]">
                      <TableCell className="py-2 pl-6 text-[13px] font-medium text-[#000000]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>
                        <HighlightedText text={item.original_filename} highlight={debouncedSearchTerm} />
                      </TableCell>
                      <TableCell className="py-2 text-[13px] text-[#000000]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>{formatFileSize(item.file_size_bytes)}</TableCell>
                      <TableCell className="py-2 text-[13px] text-[#000000]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>{formatTimestamp(item.upload_timestamp)}</TableCell>
                      <TableCell className="py-2 text-[#064372]">
                        {item.source === 'user_upload' ? <User className="h-[18px] w-[18px]" strokeWidth={1.5} /> : <FolderOpen className="h-[18px] w-[18px]" strokeWidth={1.5} />}
                      </TableCell>
                      <TableCell className="py-2">
                        <div className="flex items-center gap-2">
                          <div className={cn("w-[8px] h-[8px] rounded-full", displayStatus.dotColor)} />
                          <span className="text-[13px] font-medium text-[#000000]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>{displayStatus.text}</span>
                        </div>
                      </TableCell>
                      <TableCell className="py-2 text-center">
                        <div className="flex items-center justify-center gap-4 text-[#064372]">
                          <Link href={`/metadata/${item.id}`}>
                            <button className="p-1 hover:bg-[#E8F3F3] rounded transition-colors" title="View">
                              <Eye className="h-4 w-4" />
                            </button>
                          </Link>
                          <button
                            className="p-1 hover:bg-[#E8F3F3] rounded transition-colors disabled:opacity-30"
                            title="Details"
                            onClick={() => item.download_url && window.open(item.download_url, '_blank', 'noopener,noreferrer')}
                            disabled={!item.download_url}
                          >
                            <FileText className="h-4 w-4" />
                          </button>
                          <button
                            className="p-1 hover:bg-[#E8F3F3] rounded transition-colors disabled:opacity-30"
                            title="Download"
                            onClick={() => item.download_url && handleDownload(item.download_url, item.original_filename)}
                            disabled={!item.download_url}
                          >
                            <Download className="h-4 w-4" />
                          </button>
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })
              ) : (
                <TableRow>
                  <TableCell colSpan={6} className="h-32 text-center text-[#707070]">
                    No upload history found matching your criteria.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>

          {/* Pagination Footer */}
          <div className="px-6 py-4 bg-white border-t border-[#E0E0E0] flex flex-col md:flex-row justify-end items-center gap-[40px]">
            <div className="flex items-center gap-3">
              <span className="text-[13px] text-[#000000] font-medium" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Items per page:</span>
              <Select value={String(pageSize)} onValueChange={(value) => setPageSize(Number(value))}>
                <SelectTrigger className="w-[70px] h-[32px] border-[#D0D4DC] rounded-[4px] text-[13px] shadow-none font-medium">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="10">10</SelectItem>
                  <SelectItem value="20">20</SelectItem>
                  <SelectItem value="50">50</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center gap-[40px]">
              <span className="text-[13px] text-[#000000] font-medium" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>
                {data?.total_items ? (
                  `${(currentPage - 1) * pageSize + 1} - ${Math.min(currentPage * pageSize, data.total_items)} of ${data.total_items}`
                ) : '0 - 0 of 0'}
              </span>

              <div className="flex items-center gap-4">
                <button
                  onClick={handleFirstPage}
                  disabled={currentPage <= 1 || isLoading}
                  className="h-8 w-8 flex items-center justify-center text-[#707070] disabled:opacity-30 hover:bg-gray-100 rounded transition-colors"
                >
                  <ChevronsLeft className="h-[20px] w-[20px]" strokeWidth={1.5} />
                </button>
                <button
                  onClick={handlePreviousPage}
                  disabled={currentPage <= 1 || isLoading}
                  className="h-8 w-8 flex items-center justify-center text-[#707070] disabled:opacity-30 hover:bg-gray-100 rounded transition-colors"
                >
                  <ChevronLeft className="h-[20px] w-[20px]" strokeWidth={1.5} />
                </button>
                <button
                  onClick={handleNextPage}
                  disabled={!hasMoreData || isLoading}
                  className="h-8 w-8 flex items-center justify-center text-[#707070] disabled:opacity-30 hover:bg-gray-100 rounded transition-colors"
                >
                  <ChevronRight className="h-[20px] w-[20px]" strokeWidth={1.5} />
                </button>
                <button
                  onClick={() => { }} // Could implement if next cursors pre-fetched
                  disabled={currentPage >= totalPages || isLoading}
                  className="h-8 w-8 flex items-center justify-center text-[#707070] disabled:opacity-30 hover:bg-gray-100 rounded transition-colors"
                >
                  <ChevronsRight className="h-[20px] w-[20px]" strokeWidth={1.5} />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default withAuth(UploadHistoryPage);
