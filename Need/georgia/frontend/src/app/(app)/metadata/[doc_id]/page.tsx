'use client';

import React, { Fragment, useState, useMemo } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { useQuery, useInfiniteQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import axiosInstance from '@/lib/axiosInstance';
import withAuth from '@/components/auth/withAuth';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
    Loader2,
    AlertCircle,
    Download,
    FileText,
    ListChecks,
    ArrowLeft,
    RefreshCw,
    Filter,
    Cpu,
    Sparkles,
    Layers,
    Brain,
    CheckCircle,
    List,
    Eye,
    Upload,
    Maximize,
    Monitor,
    GitBranch,
    History,
    File,
    User
} from 'lucide-react';
import { cn } from "@/lib/utils";
import { format } from 'date-fns';
import { AxiosError } from 'axios';
import { toast } from 'sonner';
import { useAuth } from '@/context/AuthContext';
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
    Sheet,
    SheetContent,
    SheetDescription,
    SheetHeader,
    SheetTitle,
    SheetTrigger,
} from "@/components/ui/sheet";

// --- Interfaces for API Response ---
interface DocumentMetadata {
    id: string;
    original_filename: string;
    upload_timestamp: string;
    status: string;
    error_message?: string | null;
    gcs_uri: string;
    file_size_bytes: number;
    content_type: string;
    source?: string;
    user_email?: string;
    categories?: string[];
}

interface LogEntry {
    id: string;
    timestamp: string;
    level: string;
    step: string;
    message: string;
    worker_id?: string;
    original_filename?: string;
    chunk_id?: string;
    details?: Record<string, unknown>;
}

interface ChunkMetadata {
    id: string;
    original_doc_firestore_id: string;
    original_filename: string;
    chunk_gcs_uri: string;
    start_page: number;
    end_page: number;
    status: string;
    status_message?: string | null;
    error_message?: string | null;
    has_embedding?: boolean;
    ocr_confidence_score?: number | null;
    classified_chunk_document_type_label?: string | null;
    used_parser_processor_id?: string | null;
    extraction_method?: string | null;
    extraction_source?: string | null;
}

interface PaginatedChunks {
    items: ChunkMetadata[];
    next_cursor: string | null;
    limit: number;
}

interface MetadataResponse extends DocumentMetadata {
    download_url: string | null;
    user_email_address: string;
}

interface PaginatedLogs {
    items: LogEntry[];
    next_cursor: string | null;
    limit: number;
}

// --- NEW Interface for Rich Debugging Data ---
interface ChunkDetails {
    chunk_id: string;
    original_filename: string;
    full_text?: string;
    entity_metadata?: Record<string, any>;
    vector_restricts?: Array<{ namespace: string; allow_list: string[] }>;
    processing_timestamp?: string;
    worker_id?: string;
    entities?: string[];
}

interface ProcessingStats {
    total_chunks: number;
    gemini_chunks: number;
    docai_chunks: number;
    status: string;
}


// --- Helper Functions ---
const formatTimestamp = (isoString: string | undefined | null): string => {
    if (!isoString) return 'N/A';
    try {
        return format(new Date(isoString), 'MM/d/yyyy,hh:mm:ss aa');
    } catch (error) {
        console.error("Error formatting date:", error);
        return isoString;
    }
};

const formatFileSize = (bytes: number | null | undefined): string => {
    if (bytes === null || bytes === undefined || isNaN(bytes) || bytes < 0) return '0 Bytes';
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

const getStatusDisplay = (status: string | undefined | null): { text: string; dotColor: string } => {
    switch (status?.toLowerCase()) {
        case 'completed': return { text: 'Ready', dotColor: 'bg-[#4CAF50]' };
        case 'splitting':
        case 'splitting_in_progress':
        case 'pending_chunk_processing':
        case 'ocr_pending':
        case 'ocr_in_progress':
        case 'pending_vectorization':
        case 'vectorizing':
        case 'processing': return { text: 'Processing', dotColor: 'bg-[#9C27B0]' };
        case 'queued_for_splitting':
        case 'pending': return { text: 'Pending', dotColor: 'bg-[#FF9800]' };
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
            return { text: 'Failed', dotColor: 'bg-[#F44336]' };
        default: return { text: status || 'Unknown', dotColor: 'bg-gray-400' };
    }
};

const getLogLevelColor = (level: string | undefined | null): string => {
    switch (level?.toUpperCase()) {
        case 'DEBUG': return 'text-gray-500';
        case 'INFO': return 'text-blue-600';
        case 'WARNING': return 'text-yellow-600';
        case 'ERROR': return 'text-red-600';
        case 'CRITICAL': return 'text-red-800 font-bold';
        default: return 'text-gray-500';
    }
};

// --- NEW Fetch Functions for Separate Endpoints ---

const fetchMetadata = async (docId: string): Promise<MetadataResponse> => {
    if (!docId) throw new Error("Document ID is required for metadata.");
    const response = await axiosInstance.get(`/docs/${docId}/metadata`);
    return response.data;
};

const fetchChunks = async ({ pageParam, queryKey }: { pageParam: unknown, queryKey: readonly unknown[] }): Promise<PaginatedChunks> => {
    const [, docId] = queryKey as [string, string];
    if (!docId) throw new Error("Document ID is required for chunks.");
    const params = new URLSearchParams();
    params.append('limit', '20');
    const cursor = typeof pageParam === 'string' ? pageParam : undefined;
    if (cursor) {
        params.append('start_after', cursor);
    }
    const response = await axiosInstance.get(`/docs/${docId}/chunks?${params.toString()}`);
    return response.data;
};

const fetchLogs = async ({ pageParam, queryKey }: { pageParam: unknown, queryKey: readonly unknown[] }): Promise<PaginatedLogs> => {
    const [, docId, logLevel] = queryKey as [string, string, string];
    if (!docId) throw new Error("Document ID is required for logs.");

    const params = new URLSearchParams();
    params.append('limit', '50');

    if (logLevel && logLevel !== 'All Levels') {
        params.append('level', logLevel);
    }

    const cursor = typeof pageParam === 'string' ? pageParam : undefined;
    if (cursor) {
        params.append('start_after', cursor);
    }

    const response = await axiosInstance.get(`/docs/${docId}/logs?${params.toString()}`);
    return response.data;
};

const fetchProcessingStats = async (docId: string): Promise<ProcessingStats> => {
    if (!docId) throw new Error("Document ID is required for processing stats.");
    const response = await axiosInstance.get(`/docs/${docId}/processing-stats`);
    return response.data;
};

const fetchChunkDetails = async (chunkId: string): Promise<ChunkDetails> => {
    if (!chunkId) throw new Error("Chunk ID is required.");
    const response = await axiosInstance.get(`/docs/chunks/${chunkId}/details`);
    return response.data;
};

// --- Chunk Details Panel Component ---
function ChunkDetailsPanel({ chunkId, docId }: { chunkId: string, docId: string }) {
    const { data: details, isLoading: isLoadingDetails, isError: isErrorDetails, error: errorDetails } = useQuery<ChunkDetails, Error>({
        queryKey: ['chunkDetails', chunkId],
        queryFn: () => fetchChunkDetails(chunkId),
        enabled: !!chunkId,
    });

    const { data: logsData, isLoading: isLoadingLogs, isError: isErrorLogs } = useQuery<PaginatedLogs, Error>({
        queryKey: ['chunkLogs', chunkId],
        queryFn: async () => {
            const response = await axiosInstance.get(`/docs/${docId}/logs?limit=50&chunk_id=${chunkId}`);
            return response.data;
        },
        enabled: !!chunkId && !!docId,
    });

    const queryClient = useQueryClient();
    const [reprocessChunkError, setReprocessChunkError] = useState<string | null>(null);

    const reprocessChunkMutation = useMutation<{ data: { msg: string } }, AxiosError, string>({
        mutationFn: (cId: string) => {
            setReprocessChunkError(null);
            return axiosInstance.post(`/docs/chunks/${cId}/reprocess`);
        },
        onSuccess: () => {
            toast.success('Chunk reprocessing initiated!');
            setReprocessChunkError(null);
            // Invalidate relevant queries to refresh UI
            queryClient.invalidateQueries({ queryKey: ['chunkDetails', chunkId] });
            queryClient.invalidateQueries({ queryKey: ['documentChunks', docId] });
            queryClient.invalidateQueries({ queryKey: ['documentLogs', docId] });
        },
        onError: (error: AxiosError) => {
            let errorMsg = error.message || 'Failed to initiate reprocessing.';
            if (error.response?.data && typeof error.response.data === 'object' && 'msg' in error.response.data) {
                errorMsg = (error.response.data as { msg?: string }).msg || errorMsg;
            }
            toast.error(`Error: ${errorMsg}`);
            setReprocessChunkError(errorMsg);
        },
    });

    const [showReprocessChunkDialog, setShowReprocessChunkDialog] = useState(false);

    const handleReprocessChunk = () => {
        if (chunkId) {
            setShowReprocessChunkDialog(true);
        }
    };

    const confirmReprocessChunk = () => {
        if (chunkId) {
            reprocessChunkMutation.mutate(chunkId);
        }
        setShowReprocessChunkDialog(false);
    };

    if (isLoadingDetails) {
        return (
            <div className="flex h-full items-center justify-center">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
        );
    }

    if (isErrorDetails) {
        return (
            <div className="p-4 text-red-600">
                <p className="font-semibold">Failed to load details.</p>
                <p className="text-sm">{errorDetails?.message}</p>
            </div>
        );
    }

    if (!details) {
        return <div className="p-4">No details found.</div>;
    }

    return (
        <div className="space-y-6 py-6 h-full overflow-y-auto pr-2">
            <SheetHeader>
                <SheetTitle>Chunk Details</SheetTitle>
                <SheetDescription>
                    Detailed inspection for Chunk ID: <span className="font-mono text-xs break-all">{chunkId}</span>
                </SheetDescription>
                <div className="pt-2 flex justify-end">
                    <Button
                        size="sm"
                        variant="outline"
                        onClick={handleReprocessChunk}
                        disabled={reprocessChunkMutation.isPending}
                        className="text-xs h-8"
                    >
                        {reprocessChunkMutation.isPending ? (
                            <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
                        ) : (
                            <RefreshCw className="mr-2 h-3.5 w-3.5" />
                        )}
                        Reprocess Chunk
                    </Button>
                </div>
            </SheetHeader>

            {/* Entity Metadata Section */}
            <div>
                <h3 className="mb-2 text-sm font-semibold tracking-wide text-foreground uppercase">Extracted Entities</h3>
                {details.entity_metadata && Object.keys(details.entity_metadata).length > 0 ? (
                    <div className="rounded-md border bg-muted/30 p-4">
                        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                            {Object.entries(details.entity_metadata).map(([key, value]) => (
                                <div key={key} className="space-y-1">
                                    <p className="text-xs font-medium text-muted-foreground break-all">{key}</p>
                                    <p className="text-sm font-medium break-words">{String(value)}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                ) : (
                    <p className="text-sm text-muted-foreground italic">No specific entities extracted.</p>
                )}
            </div>

            {/* Searchable Entities (Deterministic) */}
            <div>
                <h3 className="mb-2 text-sm font-semibold tracking-wide text-foreground uppercase">Searchable Entities</h3>
                {details.entities && details.entities.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                        {details.entities.map((entity, idx) => (
                            <span key={idx} className="inline-flex items-center rounded-md border border-purple-200 bg-purple-50 px-2 py-1 text-xs font-medium text-purple-700 dark:border-purple-800 dark:bg-purple-900/30 dark:text-purple-300">
                                {entity}
                            </span>
                        ))}
                    </div>
                ) : (
                    <p className="text-sm text-muted-foreground italic">No searchable entities indexed.</p>
                )}
            </div>

            {/* Vector Namespaces */}
            <div>
                <h3 className="mb-2 text-sm font-semibold tracking-wide text-foreground uppercase">Vector Restrictions</h3>
                {details.vector_restricts && details.vector_restricts.length > 0 ? (
                    <div className="space-y-2">
                        {details.vector_restricts.map((restrict, idx) => (
                            <div key={idx} className="flex items-center gap-2 rounded-md border border-blue-200 bg-blue-50 px-3 py-2 text-sm text-blue-900 dark:border-blue-800 dark:bg-blue-950 dark:text-blue-100">
                                <span className="font-mono font-bold">{restrict.namespace}:</span>
                                <span>{restrict.allow_list.join(', ')}</span>
                            </div>
                        ))}
                    </div>
                ) : (
                    <p className="text-sm text-muted-foreground italic">No vector restrictions applied.</p>
                )}
            </div>

            {/* Processing History / Logs */}
            <div>
                <h3 className="mb-2 text-sm font-semibold tracking-wide text-foreground uppercase">Processing History</h3>
                {isLoadingLogs ? (
                    <p className="text-sm text-muted-foreground">Loading logs...</p>
                ) : logsData?.items && logsData.items.length > 0 ? (
                    <div className="rounded-md border text-xs">
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead className="h-8">Time</TableHead>
                                    <TableHead className="h-8">Step</TableHead>
                                    <TableHead className="h-8">Message</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {logsData.items.map((log) => (
                                    <TableRow key={log.id}>
                                        <TableCell className="py-2 whitespace-nowrap text-[10px] text-muted-foreground">
                                            {formatTimestamp(log.timestamp) || 'N/A'}
                                        </TableCell>
                                        <TableCell className="py-2 font-medium">{log.step || '-'}</TableCell>
                                        <TableCell className="py-2 break-all">{log.message}</TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </div>
                ) : (
                    <p className="text-sm text-muted-foreground italic">No logs found for this chunk.</p>
                )}
            </div>

            {/* Full Text Content */}
            <div className="flex-1">
                <h3 className="mb-2 text-sm font-semibold tracking-wide text-foreground uppercase">Extracted Text</h3>
                <div className="rounded-md border bg-slate-50 dark:bg-slate-900 p-4 font-mono text-xs leading-relaxed max-h-[300px] overflow-y-auto whitespace-pre-wrap">
                    {details.full_text || <span className="italic text-muted-foreground">No text content available.</span>}
                </div>
            </div>

            <div className="border-t pt-4">
                <p className="text-xs text-muted-foreground">
                    Processed at: {details.processing_timestamp ? formatTimestamp(details.processing_timestamp) : 'N/A'} <br />
                    Worker ID: {details.worker_id || 'N/A'}
                </p>
            </div>

            <AlertDialog open={showReprocessChunkDialog} onOpenChange={setShowReprocessChunkDialog}>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Confirm Chunk Reprocess</AlertDialogTitle>
                        <AlertDialogDescription>
                            Are you sure you want to reprocess this specific chunk?
                            This will reset its status and re-run classification, OCR, entity extraction, and vectorization for this section only.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel disabled={reprocessChunkMutation.isPending}>Cancel</AlertDialogCancel>
                        <AlertDialogAction onClick={confirmReprocessChunk} disabled={reprocessChunkMutation.isPending}>
                            {reprocessChunkMutation.isPending ? (
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            ) : null}
                            Confirm
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>
        </div>
    );
}


// --- Page Component ---
function DocumentDetailsPage() {
    const params = useParams();
    const router = useRouter();
    const queryClient = useQueryClient();
    const docId = params.doc_id as string;
    const [reprocessError, setReprocessError] = useState<string | null>(null);
    const [forceReprocessError, setForceReprocessError] = useState<string | null>(null);
    const [showReprocessConfirmDialog, setShowReprocessConfirmDialog] = useState(false); // State for confirmation dialog
    const [selectedStatusFilter, setSelectedStatusFilter] = useState<string>('All');
    const [selectedLogLevelFilter, setSelectedLogLevelFilter] = useState<string>('All Levels');
    const [isRefreshing, setIsRefreshing] = useState(false);

    const { user } = useAuth();
    const isSuperAdmin = user?.role === 'superadmin';

    const { data: metadata, isLoading: isLoadingMetadata, error: metadataError, isError: isMetadataError, isFetching: isFetchingMetadata } = useQuery<MetadataResponse, Error>({
        queryKey: ['documentMetadata', docId],
        queryFn: () => fetchMetadata(docId),
        enabled: !!docId,
        staleTime: 5 * 60 * 1000,
        refetchInterval: (query) => {
            const currentData = query.state.data;
            const status = currentData?.status?.toLowerCase();
            return status === 'processing' || status === 'pending' ? 10000 : false;
        },
    });

    const {
        data: chunksData,
        fetchNextPage: fetchNextChunksPage,
        hasNextPage: hasNextChunksPage,
        isFetchingNextPage: isFetchingNextChunks,
        isLoading: isLoadingChunks,
        isFetching: isFetchingChunks,
        error: chunksError,
        isError: isChunksError,
    } = useInfiniteQuery<PaginatedChunks, Error>({
        queryKey: ['documentChunks', docId] as const,
        queryFn: fetchChunks,
        getNextPageParam: (lastPage) => lastPage.next_cursor || undefined,
        initialPageParam: undefined,
        enabled: !!docId,
    });

    const {
        data: logsData,
        fetchNextPage: fetchNextLogsPage,
        hasNextPage: hasNextLogsPage,
        isFetchingNextPage: isFetchingNextLogs,
        isLoading: isLoadingLogs,
        isFetching: isFetchingLogs,
        error: logsError,
        isError: isLogsError,
    } = useInfiniteQuery<PaginatedLogs, Error>({
        queryKey: ['documentLogs', docId, selectedLogLevelFilter] as const,
        queryFn: fetchLogs,
        getNextPageParam: (lastPage) => lastPage.next_cursor || undefined,
        initialPageParam: undefined,
        enabled: !!docId,
    });

    const { data: stats, isLoading: isLoadingStats } = useQuery<ProcessingStats, Error>({
        queryKey: ['processingStats', docId],
        queryFn: () => fetchProcessingStats(docId),
        enabled: !!docId,
        refetchInterval: (query) => {
            const currentData = query.state.data;
            // If total chunks is 0 or less than what might be expected, maybe keep polling?
            // Actually metadata status is a better indicator
            return metadata?.status?.toLowerCase() === 'processing' ? 10000 : false;
        },
    });

    const allChunks = useMemo(() => {
        return chunksData?.pages?.flatMap(page => page.items ?? []) ?? [];
    }, [chunksData]);

    const allLogs = logsData?.pages?.flatMap(page => page.items ?? []) ?? [];
    const sortedLogs = allLogs.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

    const filteredChunks = useMemo(() => {
        if (selectedStatusFilter === 'All') {
            return allChunks;
        }
        return allChunks.filter(chunk => {
            const statusInfo = getStatusDisplay(chunk.status);
            if (selectedStatusFilter === 'Failed') {
                return statusInfo.dotColor === 'bg-[#F44336]';
            }
            return statusInfo.text === selectedStatusFilter;
        });
    }, [allChunks, selectedStatusFilter]);


    const handleDownload = () => {
        if (metadata?.download_url) {
            window.open(metadata.download_url, '_blank');
        } else {
            console.error("Download URL not available.");
        }
    };

    const handleRefresh = async () => {
        setIsRefreshing(true);
        try {
            await queryClient.invalidateQueries({ queryKey: ['documentMetadata', docId] });
            await queryClient.invalidateQueries({ queryKey: ['documentChunks', docId] });
            await queryClient.invalidateQueries({ queryKey: ['documentLogs', docId, selectedLogLevelFilter] });
            await queryClient.invalidateQueries({ queryKey: ['processingStats', docId] });
        } catch (error) {
            console.error("Failed to refresh data:", error);
        } finally {
            setIsRefreshing(false);
        }
    };

    const reprocessMutation = useMutation<{ data: { msg: string } }, AxiosError, string>({
        mutationFn: (documentId: string) => {
            setReprocessError(null);
            return axiosInstance.post(`/docs/${documentId}/reprocess`);
        },
        onSuccess: () => {
            setReprocessError(null);
            // queryClient.invalidateQueries({ queryKey: ['documentMetadata', docId] });
            handleRefresh(); // Refresh all data after successful reprocessing

        },
        onError: (error: AxiosError) => {
            let errorMsg = error.message || 'Failed to initiate reprocessing.';
            if (error.response?.data && typeof error.response.data === 'object' && 'msg' in error.response.data) {
                errorMsg = (error.response.data as { msg?: string }).msg || errorMsg;
            }
            setReprocessError(errorMsg);
        },
    });

    const handleReprocess = () => {
        setShowReprocessConfirmDialog(true); // Show confirmation dialog
    };

    const confirmReprocess = () => {
        if (docId) {
            reprocessMutation.mutate(docId);
        }
        setShowReprocessConfirmDialog(false); // Close dialog
    };

    const forceReprocessMutation = useMutation<{ data: { msg: string } }, AxiosError, string>({
        mutationFn: (documentId: string) => {
            setForceReprocessError(null);
            return axiosInstance.post(`/docs/${documentId}/force-reprocess`);
        },
        onSuccess: () => {
            toast.success('Force reprocessing initiated successfully!');
            setForceReprocessError(null);
            queryClient.invalidateQueries({ queryKey: ['documentMetadata', docId] });
        },
        onError: (error: AxiosError) => {
            let errorMsg = error.message || 'Failed to initiate force reprocessing.';
            if (error.response?.data && typeof error.response.data === 'object' && 'msg' in error.response.data) {
                errorMsg = (error.response.data as { msg?: string }).msg || errorMsg;
            }
            toast.error(`Error: ${errorMsg}`);
            setForceReprocessError(errorMsg);
        },
    });

    const handleForceReprocess = () => {
        if (docId) {
            forceReprocessMutation.mutate(docId);
        }
    };

    const isAnyFetching = isFetchingMetadata || isFetchingChunks || isFetchingLogs;
    const isInitialLoading = isLoadingMetadata || isLoadingChunks || isLoadingLogs;
    const isError = isMetadataError || isChunksError || isLogsError;
    const error = metadataError || chunksError || logsError;

    if (isInitialLoading && !metadata && allChunks.length === 0 && allLogs.length === 0) {
        return (
            <div className="flex items-center justify-center h-screen">
                <Loader2 className="h-16 w-16 animate-spin text-muted-foreground" />
            </div>
        );
    }

    if (isError && !metadata) {
        return (
            <div className="container mx-auto p-8">
                <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error Loading Document Details</AlertTitle>
                    <AlertDescription>
                        {error?.message ?? 'An unknown error occurred.'}
                    </AlertDescription>
                </Alert>
            </div>
        );
    }

    if (!isLoadingMetadata && !metadata) {
        return (
            <div className="container mx-auto p-8 text-center text-muted-foreground">
                Document metadata not found or failed to load.
            </div>
        );
    }

    const overallStatus = getStatusDisplay(metadata?.status);
    const totalLogCount = logsData?.pages?.reduce((acc, page) => acc + (page?.items?.length ?? 0), 0) ?? 0;

    return (
        <div className="flex-1 bg-[#ffffff] p-8 overflow-y-auto">
            <div className="max-w-7xl mx-auto space-y-8">
                {/* Header Section */}
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                    <div className="flex items-center gap-4">
                        <div onClick={() => router.push('/history')} className="h-10 w-10 cursor-pointer flex justify-center items-center bg-[#E8F3F3] rounded-full text-[#255c5d]">
                            <ArrowLeft className="h-5 w-5" />
                        </div>
                        <div className="flex flex-col">
                            <h1
                                className="text-2xl font-bold"
                                style={{ fontFamily: 'var(--unnamed-font-family-georgia)' }}
                            >
                                {metadata?.original_filename || 'Document Details'}
                            </h1>
                            <p className="text-xs text-[#707070]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>
                                Document ID: {metadata?.id}
                            </p>
                        </div>
                    </div>
                    <div className="flex items-center gap-3">
                        <Button
                            variant="outline"
                            size="icon"
                            onClick={handleDownload}
                            disabled={!metadata?.download_url}
                            className="h-10 w-10 bg-white border-[#255c5d] text-[#255c5d] rounded-xl hover:bg-[#E8F3F3]"
                        >
                            <Download className="h-5 w-5" />
                        </Button>
                        {
                            isSuperAdmin && <Button
                                variant="outline"
                                size="icon"
                                onClick={handleRefresh}
                                disabled={isRefreshing || isAnyFetching}
                                className="h-10 w-10 bg-white border-[#255c5d] text-[#255c5d] rounded-xl hover:bg-[#E8F3F3]"
                            >
                                {isRefreshing || isAnyFetching ? (
                                    <Loader2 className="h-5 w-5 animate-spin" />
                                ) : (
                                    <RefreshCw className="h-5 w-5" />
                                )}
                            </Button>
                        }

                    </div>
                </div>

                {/* Processing Statistics Cards */}
                {
                    isSuperAdmin && <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div style={{ background: '#FFFFFF', boxShadow: '0px 3px 6px #D0D4DC40', border: '2px solid #EEF4F4' }} className="rounded-[4px] px-5 py-4 flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <Layers className="h-4 w-4" />
                                <span className="font-semibold" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Total Chunks</span>
                            </div>
                            <span className="inline-flex items-center justify-center min-w-[28px] h-[24px] px-2 rounded-[4px] bg-[#57B6B2] text-xs font-bold opacity-70">{stats?.total_chunks ?? 0}</span>
                        </div>

                        <div style={{ background: '#FFFFFF', boxShadow: '0px 3px 6px #D0D4DC40', border: '2px solid #EEF4F4' }} className="rounded-[4px] px-5 py-4 flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <Brain className="h-4 w-4" />
                                <span className="font-semibold" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>LLM Extractions OCR</span>
                            </div>
                            <span className="inline-flex items-center justify-center min-w-[28px] h-[24px] px-2 rounded-[4px] bg-[#57B6B2] text-xs font-bold opacity-70">{stats?.gemini_chunks ?? 0}</span>
                        </div>

                        <div style={{ background: '#FFFFFF', boxShadow: '0px 3px 6px #D0D4DC40', border: '2px solid #EEF4F4' }} className="rounded-[4px] px-5 py-4 flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <Cpu className="h-4 w-4" />
                                <span className="font-semibold" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Legacy OCR</span>
                            </div>
                            <span className="inline-flex items-center justify-center min-w-[28px] h-[24px] px-2 rounded-[4px] bg-[#57B6B2] text-xs font-bold opacity-70">{stats?.docai_chunks ?? 0}</span>
                        </div>
                    </div>
                }


                {/* Document Information Grid */}
                {metadata && (
                    <div className="bg-white border border-[#E0E0E0] rounded-[8px] overflow-hidden shadow-sm">
                        <div style={{ background: '#EEF8F7', borderRadius: '8px 8px 0px 0px' }} className="px-6 py-3 flex flex-row items-center justify-between">
                            <span className="text-sm font-bold text-[#000000] uppercase" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Document Information</span>
                            <div className="flex items-center gap-1.5">
                                <span className="text-sm font-bold text-[#000000]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Status</span>
                                <div className={cn("w-2.5 h-2.5 rounded-full ml-1", overallStatus.dotColor)} />
                                <span className="text-sm text-[#000000]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>{overallStatus.text}</span>
                            </div>
                        </div>
                        <div className="p-6">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-y-8 gap-x-12">
                                <div className="flex items-start gap-4">
                                    <div className="mt-1 p-1.5 bg-[#E8F3F3] rounded text-[#255c5d]">
                                        <File className="h-4 w-4" />
                                    </div>
                                    <div>
                                        <p className="text-[#000000] text-sm font-medium" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Filename</p>
                                        <p className="text-sm text-[#555555] break-all">{metadata.original_filename}</p>
                                    </div>
                                </div>
                                <div className="flex items-start gap-4">
                                    <div className="mt-1 p-1.5 bg-[#E8F3F3] rounded text-[#255c5d]">
                                        <History className="h-4 w-4" />
                                    </div>
                                    <div>
                                        <p className="text-[#000000] text-sm font-medium" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Uploaded by</p>
                                        <p className="text-sm text-[#555555]">{metadata.user_email_address || 'System'}</p>
                                    </div>
                                </div>
                                <div className="flex items-start gap-4">
                                    <div className="mt-1 p-1.5 bg-[#E8F3F3] rounded text-[#255c5d]">
                                        <Upload className="h-4 w-4" />
                                    </div>
                                    <div>
                                        <p className="text-[#000000] text-sm font-medium" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Upload Date & Time</p>
                                        <p className="text-sm text-[#555555]">{formatTimestamp(metadata.upload_timestamp)}</p>
                                    </div>
                                </div>
                                <div className="flex items-start gap-4">
                                    <div className="mt-1 p-1.5 bg-[#E8F3F3] rounded text-[#255c5d]">
                                        <Maximize className="h-4 w-4" />
                                    </div>
                                    <div>
                                        <p className="text-[#000000] text-sm font-medium" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>File Size</p>
                                        <p className="text-sm text-[#555555]">{formatFileSize(metadata.file_size_bytes)}</p>
                                    </div>
                                </div>
                                <div className="flex items-start gap-4">
                                    <div className="mt-1 p-1.5 bg-[#E8F3F3] rounded text-[#255c5d]">
                                        <Monitor className="h-4 w-4" />
                                    </div>
                                    <div>
                                        <p className="text-[#000000] text-sm font-medium" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Content type</p>
                                        <p className="text-sm text-[#555555] font-mono">{metadata.content_type}</p>
                                    </div>
                                </div>
                                <div className="flex items-start gap-4">
                                    <div className="mt-1 p-1.5 bg-[#E8F3F3] rounded text-[#255c5d]">
                                        <GitBranch className="h-4 w-4" />
                                    </div>
                                    <div>
                                        <p className="text-[#000000] text-sm font-medium" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Source</p>
                                        <p className="text-sm text-[#555555]">{metadata.source || 'N/A'}</p>
                                    </div>
                                </div>
                                <div className="flex items-start gap-4">
                                    <div className="mt-1 p-1.5 bg-[#E8F3F3] rounded text-[#255c5d]">
                                        <Layers className="h-4 w-4" />
                                    </div>
                                    <div>
                                        <p className="text-[#000000] text-sm font-medium" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Document Category</p>
                                        <p className="text-sm text-[#555555]">{metadata.categories?.join(', ') || 'Other'}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Document Chunks Section */}
                {isSuperAdmin && (<div className="mt-6">
                    <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-2 mb-3">
                        <div>
                            <h2 className="text-xl font-bold text-[#000000]" style={{ fontFamily: 'var(--unnamed-font-family-georgia)' }}>Document Chunks ({filteredChunks.length})</h2>
                            <p className="text-xs text-[#707070] mt-1" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Individual sections processed from the document.</p>
                        </div>
                        <Select value={selectedStatusFilter} onValueChange={setSelectedStatusFilter}>
                            <SelectTrigger className="w-[100px] h-9 bg-white border-[#E0E0E0] rounded-[4px] text-xs" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>
                                <SelectValue placeholder="All" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="All">All</SelectItem>
                                <SelectItem value="Completed">Ready</SelectItem>
                                <SelectItem value="Processing">Processing</SelectItem>
                                <SelectItem value="Pending">Pending</SelectItem>
                                <SelectItem value="Failed">Failed</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>
                    <div className="bg-white border border-[#E0E0E0] rounded-[8px] overflow-hidden shadow-sm">
                        <div className="overflow-x-auto">
                            <Table>
                                <TableHeader>
                                    <TableRow style={{ background: '#EEF8F7' }} className="border-b border-[#E0E0E0] hover:bg-[#EEF8F7]">
                                        <TableHead className="text-[11px] font-bold text-[#707070] uppercase tracking-wider py-3 pl-6">SL</TableHead>
                                        <TableHead className="text-[11px] font-bold text-[#707070] uppercase tracking-wider py-3">Chunk ID</TableHead>
                                        <TableHead className="text-[11px] font-bold text-[#707070] uppercase tracking-wider py-3">Pages</TableHead>
                                        <TableHead className="text-[11px] font-bold text-[#707070] uppercase tracking-wider py-3">Status</TableHead>
                                        <TableHead className="text-[11px] font-bold text-[#707070] uppercase tracking-wider py-3">Confidence</TableHead>
                                        <TableHead className="text-[11px] font-bold text-[#707070] uppercase tracking-wider py-3">Processor Type</TableHead>
                                        <TableHead className="text-[11px] font-bold text-[#707070] uppercase tracking-wider py-3">Extraction Type</TableHead>
                                        <TableHead className="text-[11px] font-bold text-[#707070] uppercase tracking-wider py-3">Message</TableHead>
                                        <TableHead className="text-[11px] font-bold text-[#707070] uppercase tracking-wider py-3 text-right pr-6">Actions</TableHead>
                                    </TableRow>
                                </TableHeader>
                                <TableBody>
                                    {filteredChunks.map((chunk, index) => {
                                        const chunkStatus = getStatusDisplay(chunk.status);
                                        return (
                                            <TableRow key={chunk.id} className="border-b border-[#E0E0E0] hover:bg-gray-50 transition-colors">
                                                <TableCell className="py-4 pl-6 text-sm ">{index + 1}</TableCell>
                                                <TableCell className="py-4 text-sm max-w-[100px] truncate" title={chunk.id}>{chunk.id}</TableCell>
                                                <TableCell className="py-4 text-sm font-medium text-[#000000]">{chunk.start_page}-{chunk.end_page}</TableCell>
                                                <TableCell className="py-4">
                                                    <div className="flex items-center gap-1.5">
                                                        <div className={cn("w-2 h-2 rounded-full", chunkStatus.dotColor)} />
                                                        <span className="text-sm text-[#000000]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>{chunkStatus.text}</span>
                                                    </div>
                                                </TableCell>
                                                <TableCell className="py-4 text-sm text-[#000000]">
                                                    {typeof chunk.ocr_confidence_score === 'number'
                                                        ? `${(chunk.ocr_confidence_score * 100).toFixed(0)}%`
                                                        : '-'}
                                                </TableCell>
                                                <TableCell className="py-4 text-sm text-[#000000]">
                                                    {chunk.classified_chunk_document_type_label || 'Other'}
                                                </TableCell>
                                                <TableCell className="py-4 text-sm text-[#000000]">
                                                    {chunk.extraction_method === 'multimodal' ? 'LLM Extraction' : 'Legacy OCR'}
                                                </TableCell>
                                                <TableCell className="py-4 text-xs text-[#000000] max-w-[200px] truncate" title={chunk.status_message || 'No message'}>
                                                    {chunk.status_message || 'Chunk processing successful with per-chunk entity extraction.'}
                                                </TableCell>
                                                <TableCell className="py-4 text-right pr-6">
                                                    <div className="flex items-center justify-end gap-3 text-[#255c5d]">
                                                        <Sheet>
                                                            <SheetTrigger asChild>
                                                                <button className="p-1 hover:bg-[#E8F3F3] rounded transition-colors" title="View Details">
                                                                    <Eye className="h-4 w-4" />
                                                                </button>
                                                            </SheetTrigger>
                                                            <SheetContent className="w-[800px] sm:max-w-[800px] overflow-y-auto">
                                                                <ChunkDetailsPanel chunkId={chunk.id} docId={docId} />
                                                            </SheetContent>
                                                        </Sheet>
                                                        <button
                                                            className="p-1 hover:bg-[#E8F3F3] rounded transition-colors"
                                                            title="Download Chunk"
                                                            onClick={() => {
                                                                const fetchDownload = async () => {
                                                                    try {
                                                                        const resp = await axiosInstance.get(`/docs/chunks/${chunk.id}/download`);
                                                                        if (resp.data.signed_url) {
                                                                            window.open(resp.data.signed_url, '_blank');
                                                                        }
                                                                    } catch (e) {
                                                                        toast.error("Download failed");
                                                                    }
                                                                };
                                                                fetchDownload();
                                                            }}
                                                        >
                                                            <Download className="h-4 w-4" />
                                                        </button>
                                                    </div>
                                                </TableCell>
                                            </TableRow>
                                        );
                                    })}
                                </TableBody>
                            </Table>
                            {hasNextChunksPage && (
                                <div className="p-4 text-center border-t border-[#E0E0E0]">
                                    <button
                                        onClick={() => fetchNextChunksPage()}
                                        disabled={isFetchingNextChunks}
                                        style={{ background: '#FFFFFF', border: '2px solid #005D5E' }}
                                        className="px-6 py-2 rounded-[4px] text-[#005D5E] text-xs font-bold uppercase tracking-widest hover:bg-[#F1F8F8] transition-colors disabled:opacity-50 inline-flex items-center"
                                    >
                                        {isFetchingNextChunks ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                                        LOAD MORE CHUNKS
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                </div>)}

                {/* Processing Logs - Super Admin Only */}
                {isSuperAdmin && (
                    <div className="mt-6">
                        <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-2 mb-3">
                            <div>
                                <h2 className="text-xl font-bold text-[#000000]" style={{ fontFamily: 'var(--unnamed-font-family-georgia)' }}>Processing Logs ({totalLogCount})</h2>
                                <p className="text-xs text-[#707070] mt-1" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Detailed logs generated during the document processing pipeline.</p>
                            </div>
                            <Select value={selectedLogLevelFilter} onValueChange={setSelectedLogLevelFilter}>
                                <SelectTrigger className="w-[100px] h-9 bg-white border-[#E0E0E0] rounded-[4px] text-xs" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>
                                    <SelectValue placeholder="All" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="All Levels">All</SelectItem>
                                    <SelectItem value="DEBUG">DEBUG</SelectItem>
                                    <SelectItem value="INFO">INFO</SelectItem>
                                    <SelectItem value="WARNING">WARNING</SelectItem>
                                    <SelectItem value="ERROR">ERROR</SelectItem>
                                    <SelectItem value="CRITICAL">CRITICAL</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                        <div className="bg-white border border-[#E0E0E0] rounded-[8px] overflow-hidden shadow-sm">
                            <div className="overflow-x-auto">
                                <Table>
                                    <TableHeader>
                                        <TableRow style={{ background: '#EEF8F7' }} className="border-b border-[#E0E0E0] hover:bg-[#EEF8F7]">
                                            <TableHead className="text-[11px] font-bold text-[#707070] uppercase tracking-wider py-3 pl-6">Timestamp</TableHead>
                                            <TableHead className="text-[11px] font-bold text-[#707070] uppercase tracking-wider py-3">Level</TableHead>
                                            <TableHead className="text-[11px] font-bold text-[#707070] uppercase tracking-wider py-3">Step</TableHead>
                                            <TableHead className="text-[11px] font-bold text-[#707070] uppercase tracking-wider py-3">Message</TableHead>
                                            <TableHead className="text-[11px] font-bold text-[#707070] uppercase tracking-wider py-3">Chunk ID</TableHead>
                                            <TableHead className="text-[11px] font-bold text-[#707070] uppercase tracking-wider py-3 pr-6">Worker ID</TableHead>
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                        {sortedLogs.map((log) => (
                                            <TableRow key={log.id} className="border-b border-[#E0E0E0] hover:bg-gray-50 transition-colors">
                                                <TableCell className="py-4 pl-6 text-xs text-[#000000] whitespace-nowrap">{formatTimestamp(log.timestamp)}</TableCell>
                                                <TableCell className="py-4">
                                                    <div className="flex items-center gap-1.5">
                                                        <div className={cn("w-2 h-2 rounded-full",
                                                            log.level?.toUpperCase() === 'INFO' ? 'bg-blue-500' :
                                                                log.level?.toUpperCase() === 'WARNING' ? 'bg-yellow-500' :
                                                                    log.level?.toUpperCase() === 'ERROR' || log.level?.toUpperCase() === 'CRITICAL' ? 'bg-red-500' :
                                                                        log.level?.toUpperCase() === 'DEBUG' ? 'bg-gray-400' :
                                                                            'bg-green-500'
                                                        )} />
                                                        <span className="text-xs font-medium text-[#000000]">{log.level}</span>
                                                    </div>
                                                </TableCell>
                                                <TableCell className="py-4 text-xs font-medium text-[#000000]">{log.step || '-'}</TableCell>
                                                <TableCell className="py-4 text-xs text-[#000000] max-w-md truncate" title={log.message}>{log.message}</TableCell>
                                                <TableCell className="py-4 text-xs text-[#000000] max-w-[120px] truncate" title={log.chunk_id || '-'}>{log.chunk_id || '-'}</TableCell>
                                                <TableCell className="py-4 pr-6 text-xs text-[#000000] max-w-[140px] truncate" title={log.worker_id || '-'}>{log.worker_id || '-'}</TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                                {hasNextLogsPage && (
                                    <div className="p-4 text-center border-t border-[#E0E0E0]">
                                        <button
                                            onClick={() => fetchNextLogsPage()}
                                            disabled={isFetchingNextLogs}
                                            style={{ background: '#FFFFFF', border: '2px solid #005D5E' }}
                                            className="px-6 py-2 rounded-[4px] text-[#005D5E] text-xs font-bold uppercase tracking-widest hover:bg-[#F1F8F8] transition-colors disabled:opacity-50 inline-flex items-center"
                                        >
                                            {isFetchingNextLogs ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                                            LOAD MORE LOGS
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Reprocess Confirmation Dialog */}
            <AlertDialog open={showReprocessConfirmDialog} onOpenChange={setShowReprocessConfirmDialog}>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Confirm Reprocess</AlertDialogTitle>
                        <AlertDialogDescription>
                            Are you sure you want to re-process this document? This will restart the entire processing pipeline for this document.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel disabled={reprocessMutation.isPending}>Cancel</AlertDialogCancel>
                        <AlertDialogAction onClick={confirmReprocess} disabled={reprocessMutation.isPending} className="bg-[#255c5d] hover:bg-[#1E4E55]">
                            {reprocessMutation.isPending ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                            Confirm
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>
        </div>
    );
}

export default withAuth(DocumentDetailsPage);
