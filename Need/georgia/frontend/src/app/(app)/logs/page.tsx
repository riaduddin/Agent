"use client";

import withAuth from "@/components/auth/withAuth";
import React, { useState, useMemo, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import axiosInstance from '@/lib/axiosInstance';
import { format } from 'date-fns';
import { useRouter } from 'next/navigation';
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Loader2, AlertCircle, ChevronLeft, ChevronRight, ArrowLeft, Download } from 'lucide-react';
import { cn } from "@/lib/utils";
import { useDebounce } from "@/hooks/useDebounce";

interface LogEntry {
    id: string;
    timestamp: string;
    level: 'INFO' | 'WARNING' | 'ERROR' | 'DEBUG' | 'CRITICAL';
    message: string;
    step?: string;
    document_id?: string;
    chunk_id?: string;
    worker_id?: string;
    original_filename?: string;
    details?: Record<string, unknown>;
}

interface LogsApiResponse {
    logs: LogEntry[];
    next_cursor: string | null;
    limit: number;
}

const fetchLogs = async (
    limit: number,
    startAfter: string | null,
    level: string,
    step: string,
    documentName: string,
    workerId: string
): Promise<LogsApiResponse> => {
    const params = new URLSearchParams();
    params.append('limit', String(limit));
    if (startAfter) params.append('start_after', startAfter);
    if (level && level !== 'all') params.append('level', level);
    if (step) params.append('step', step);
    if (documentName) params.append('original_filename', documentName);
    if (workerId) params.append('worker_id', workerId);

    const response = await axiosInstance.get('/logs', { params });
    return response.data;
};

const formatTimestamp = (isoString: string): string => {
    try {
        return format(new Date(isoString), 'Pp');
    } catch {
        return isoString;
    }
};

const getLogLevelClass = (level: string): string => {
    switch (level?.toUpperCase()) {
        case 'ERROR': return 'text-red-600 font-semibold';
        case 'CRITICAL': return 'text-red-800 font-bold';
        case 'WARNING': return 'text-yellow-600 font-semibold';
        case 'INFO': return 'text-blue-600';
        default: return 'text-gray-500';
    }
};

const LOG_LEVELS = ['INFO', 'WARNING', 'ERROR', 'CRITICAL', 'DEBUG'];

function ProcessingLogsPage() {
    const [pageSize] = useState(50);
    const [currentPage, setCurrentPage] = useState(1);
    const [pageCursors, setPageCursors] = useState<(string | null)[]>([null]);
    const router = useRouter();
    // const queryClient = useQueryClient();

    const [levelFilter, setLevelFilter] = useState('all');
    const [stepFilterInput, setStepFilterInput] = useState('');
    const [documentNameFilterInput, setDocumentNameFilterInput] = useState('');
    const [workerIdFilterInput, setWorkerIdFilterInput] = useState('');

    const debouncedStepFilter = useDebounce(stepFilterInput, 500);
    const debouncedDocumentNameFilter = useDebounce(documentNameFilterInput, 500);
    const debouncedWorkerIdFilter = useDebounce(workerIdFilterInput, 500);

    useEffect(() => {
        setCurrentPage(1);
        setPageCursors([null]);
    }, [levelFilter, debouncedStepFilter, debouncedDocumentNameFilter, debouncedWorkerIdFilter]);

    const currentCursor = useMemo(() => pageCursors[currentPage - 1], [pageCursors, currentPage]);
    const queryKey = useMemo(() => [
        'processingLogs',
        pageSize,
        currentCursor,
        levelFilter,
        debouncedStepFilter,
        debouncedDocumentNameFilter,
        debouncedWorkerIdFilter
    ], [pageSize, currentCursor, levelFilter, debouncedStepFilter, debouncedDocumentNameFilter, debouncedWorkerIdFilter]);

    const { data, isLoading, error, isError, isFetching } = useQuery<LogsApiResponse>({
        queryKey,
        queryFn: () => fetchLogs(
            pageSize,
            currentCursor,
            levelFilter,
            debouncedStepFilter,
            debouncedDocumentNameFilter,
            debouncedWorkerIdFilter
        ),
        refetchInterval: 3000,
        refetchOnWindowFocus: true,
    });

    const handleNextPage = () => {
        if (data?.next_cursor) {
            if (pageCursors.length === currentPage) {
                setPageCursors(prev => [...prev, data.next_cursor ?? null]);
            }
            setCurrentPage(prev => prev + 1);
        }
    };

    const handlePreviousPage = () => {
        if (currentPage > 1) {
            setCurrentPage(prev => prev - 1);
        }
    };

    const hasMoreData = !!data?.next_cursor && data?.logs?.length === pageSize;

    const handleExport = () => {
        const params = new URLSearchParams();
        if (levelFilter && levelFilter !== 'all') params.append('level', levelFilter);
        if (debouncedStepFilter) params.append('step', debouncedStepFilter);
        if (debouncedDocumentNameFilter) params.append('original_filename', debouncedDocumentNameFilter);
        if (debouncedWorkerIdFilter) params.append('worker_id', debouncedWorkerIdFilter);

        const exportUrl = `${axiosInstance.defaults.baseURL}/logs/export?${params.toString()}`;
        window.open(exportUrl, '_blank');
    };

    return (
        <div className="flex-1 bg-[#F8F8F8] p-8 overflow-y-auto">
            <div className="max-w-7xl mx-auto space-y-8">
                {/* Header Section */}
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
                    <div className="flex items-center gap-4">
                        <Button
                            variant="outline"
                            size="icon"
                            onClick={() => router.back()}
                            className="bg-white border-[#E0E0E0] text-[#255c5d] hover:bg-[#E8F3F3] rounded-full h-10 w-10 transition-colors"
                        >
                            <ArrowLeft className="h-5 w-5" />
                        </Button>
                        <div>
                            <h1
                                className="text-[28px] font-bold text-[#255c5d] leading-tight"
                                style={{ fontFamily: 'var(--unnamed-font-family-georgia)' }}
                            >
                                Processing Logs
                            </h1>
                            <p
                                className="text-xs text-[#707070] mt-1"
                                style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
                            >
                                Real-time logs from the document processing pipeline
                            </p>
                        </div>
                    </div>

                    <div className="flex flex-wrap items-center gap-3">
                        {/* Level Filter */}
                        <div className="flex items-center gap-2 bg-white border border-[#E0E0E0] rounded-[4px] px-3 h-10">
                            <span className="text-[10px] font-bold text-[#707070] uppercase tracking-wider" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Level</span>
                            <Select value={levelFilter} onValueChange={setLevelFilter}>
                                <SelectTrigger className="border-none focus:ring-0 shadow-none h-auto p-0 text-xs font-medium text-[#255c5d] bg-transparent w-[100px]">
                                    <SelectValue placeholder="All" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="all">All Levels</SelectItem>
                                    {LOG_LEVELS.map(level => (
                                        <SelectItem key={level} value={level}>{level}</SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>

                        {/* Doc Name Filter */}
                        <div className="flex items-center gap-2 bg-white border border-[#E0E0E0] rounded-[4px] px-3 h-10 focus-within:border-[#255c5d] transition-colors">
                            <span className="text-[10px] font-bold text-[#707070] uppercase tracking-wider" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Doc</span>
                            <Input
                                type="text"
                                placeholder="..."
                                value={documentNameFilterInput}
                                onChange={(e) => setDocumentNameFilterInput(e.target.value)}
                                className="border-none focus-visible:ring-0 shadow-none h-auto p-0 text-xs font-medium text-[#255c5d] bg-transparent w-[100px]"
                            />
                        </div>

                        <Button
                            variant="outline"
                            onClick={handleExport}
                            className="bg-white border-[#255c5d] text-[#255c5d] font-bold uppercase text-[10px] tracking-widest rounded-[4px] hover:bg-[#E8F3F3] h-10 px-4 transition-colors"
                            style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
                        >
                            <Download className="h-3 w-3 mr-2" />
                            Export
                        </Button>

                        <Button
                            variant="ghost"
                            onClick={() => {
                                setLevelFilter('all');
                                setStepFilterInput('');
                                setDocumentNameFilterInput('');
                                setWorkerIdFilterInput('');
                            }}
                            className="text-[#707070] font-bold uppercase text-[10px] tracking-widest hover:text-[#255c5d] h-10 transition-colors"
                            style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
                        >
                            Reset
                        </Button>

                        {isFetching && !isLoading && <Loader2 className="h-4 w-4 animate-spin text-[#255c5d]" />}
                    </div>
                </div>

                {/* Logs Table Card */}
                <div className="bg-white border border-[#E0E0E0] rounded-[8px] overflow-hidden shadow-sm">
                    <div className="bg-[#F1F8F8] px-6 py-4 border-b border-[#E0E0E0] flex justify-between items-center">
                        <h2 className="text-[11px] font-bold text-[#255c5d] uppercase tracking-[0.2em]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>System Processing Logs</h2>
                        <span className="text-[10px] text-[#255c5d]/60 font-medium italic">Refreshes every 3s</span>
                    </div>

                    <div className="overflow-x-auto">
                        <Table className="table-fixed w-full">
                            <TableHeader className="bg-gray-50/50">
                                <TableRow className="border-b border-[#E0E0E0] hover:bg-transparent">
                                    <TableHead className="w-[180px] text-[11px] font-bold text-[#707070] uppercase tracking-wider py-4 pl-6">Timestamp</TableHead>
                                    <TableHead className="w-[100px] text-[11px] font-bold text-[#707070] uppercase tracking-wider py-4">Level</TableHead>
                                    <TableHead className="hidden md:table-cell w-[180px] text-[11px] font-bold text-[#707070] uppercase tracking-wider py-4">Step</TableHead>
                                    <TableHead className="text-[11px] font-bold text-[#707070] uppercase tracking-wider py-4">Detailed Message</TableHead>
                                    <TableHead className="hidden md:table-cell w-[150px] text-[11px] font-bold text-[#707070] uppercase tracking-wider py-4 pr-6">Document</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {isLoading ? (
                                    <TableRow>
                                        <TableCell colSpan={5} className="py-20 text-center">
                                            <div className="flex flex-col items-center gap-3">
                                                <Loader2 className="h-8 w-8 animate-spin text-[#255c5d]/20" />
                                                <span className="text-xs text-[#707070] font-medium italic">Synchronizing Logs...</span>
                                            </div>
                                        </TableCell>
                                    </TableRow>
                                ) : isError ? (
                                    <TableRow>
                                        <TableCell colSpan={5} className="py-20 text-center">
                                            <div className="flex flex-col items-center gap-3">
                                                <AlertCircle className="h-8 w-8 text-red-400" />
                                                <span className="text-xs text-red-600 font-medium">Failed to reconcile system logs</span>
                                            </div>
                                        </TableCell>
                                    </TableRow>
                                ) : data?.logs && data.logs.length > 0 ? (
                                    data.logs.map((log) => (
                                        <TableRow key={log.id} className="border-b border-[#E0E0E0] hover:bg-gray-50 transition-colors">
                                            <TableCell className="py-4 pl-6">
                                                <span className="text-[12px] font-medium text-[#707070] font-mono">{formatTimestamp(log.timestamp)}</span>
                                            </TableCell>
                                            <TableCell className="py-4">
                                                <span className={cn(
                                                    "text-[10px] font-bold uppercase tracking-tight px-2 py-0.5 rounded-full border",
                                                    log.level === 'ERROR' || log.level === 'CRITICAL'
                                                        ? "bg-red-50 text-red-600 border-red-200"
                                                        : log.level === 'WARNING'
                                                            ? "bg-yellow-50 text-yellow-600 border-yellow-200"
                                                            : "bg-blue-50 text-blue-600 border-blue-200"
                                                )}>
                                                    {log.level}
                                                </span>
                                            </TableCell>
                                            <TableCell className="hidden md:table-cell py-4">
                                                <span className="text-xs font-medium text-[#255c5d]">{log.step || '-'}</span>
                                            </TableCell>
                                            <TableCell className="py-4 pr-4">
                                                <p className="text-xs text-[#000000] leading-relaxed max-w-2xl">{log.message}</p>
                                            </TableCell>
                                            <TableCell className="hidden md:table-cell py-4 pr-6">
                                                {log.original_filename ? (
                                                    <div className="flex items-center gap-1.5 text-xs font-medium text-[#707070] max-w-[140px] truncate" title={log.original_filename}>
                                                        <span className="opacity-40">#</span>
                                                        {log.original_filename}
                                                    </div>
                                                ) : <span className="text-xs text-[#E0E0E0]">-</span>}
                                            </TableCell>
                                        </TableRow>
                                    ))
                                ) : (
                                    <TableRow>
                                        <TableCell colSpan={5} className="py-20 text-center">
                                            <p className="text-sm text-[#707070] italic">No log entries found matching criteria.</p>
                                        </TableCell>
                                    </TableRow>
                                )}
                            </TableBody>
                        </Table>
                    </div>

                    {/* Footer / Pagination */}
                    {(data?.logs && data.logs.length > 0) && (
                        <div className="bg-gray-50/50 px-6 py-4 flex justify-between items-center border-t border-[#E0E0E0]">
                            <span className="text-[10px] font-bold text-[#707070] uppercase tracking-widest" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>
                                Page {currentPage}
                            </span>
                            <div className="flex gap-2">
                                <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={handlePreviousPage}
                                    disabled={currentPage <= 1 || isLoading || isFetching}
                                    className="h-8 rounded-[4px] border-[#E0E0E0] uppercase text-[10px] font-bold tracking-widest"
                                >
                                    <ChevronLeft className="h-3 w-3 mr-1" /> Prev
                                </Button>
                                <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={handleNextPage}
                                    disabled={!hasMoreData || isLoading || isFetching}
                                    className="h-8 rounded-[4px] border-[#E0E0E0] uppercase text-[10px] font-bold tracking-widest"
                                >
                                    Next <ChevronRight className="h-3 w-3 ml-1" />
                                </Button>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

export default withAuth(ProcessingLogsPage);
