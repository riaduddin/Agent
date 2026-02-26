"use client";

import withAuth from "@/components/auth/withAuth";
import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import axiosInstance from '@/lib/axiosInstance';
import { format } from 'date-fns';
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { History, Loader2, User, Clock, CheckCircle, XCircle, AlertCircle, ArrowLeft, RefreshCw } from 'lucide-react';
import Link from 'next/link';
import { useAuth } from "@/context/AuthContext";
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';
import { toast } from 'sonner';
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

interface LegacyRun {
    run_id: string;
    start_time: string;
    end_time: string | null;
    trigger_source: string;
    initiated_by: string;
    status: string;
    docs_found: number;
    docs_queued: number;
    docs_skipped: number;
    skip_aged_chunks: boolean;
    max_age_days: number | null;
    errors: string[];
}

interface HistoryResponse {
    runs: LegacyRun[];
    next_cursor: string | null;
    limit: number;
}

const fetchHistory = async (limit: number): Promise<HistoryResponse> => {
    const response = await axiosInstance.get('/docs/legacy-reprocess-history', {
        params: { limit }
    });
    return response.data;
};

const getStatusBadge = (status: string) => {
    switch (status) {
        case 'completed':
            return <Badge className="bg-green-500 hover:bg-green-600"><CheckCircle className="h-3 w-3 mr-1" />Completed</Badge>;
        case 'scanning':
        case 'queuing':
            return <Badge className="bg-yellow-500 hover:bg-yellow-600"><Clock className="h-3 w-3 mr-1" />In Progress</Badge>;
        case 'error':
            return <Badge className="bg-red-500 hover:bg-red-600"><XCircle className="h-3 w-3 mr-1" />Error</Badge>;
        default:
            return <Badge className="bg-gray-500 hover:bg-gray-600"><AlertCircle className="h-3 w-3 mr-1" />{status}</Badge>;
    }
};

const getTriggerBadge = (source: string) => {
    return source === "MANUAL"
        ? <Badge variant="outline" className="bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800"><User className="h-3 w-3 mr-1" />Manual</Badge>
        : <Badge variant="outline" className="bg-purple-50 dark:bg-purple-950 border-purple-200 dark:border-purple-800"><Clock className="h-3 w-3 mr-1" />Scheduled</Badge>;
};

function LegacyReprocessHistoryPage() {
    const { user } = useAuth();
    const router = useRouter();
    const isSuperAdmin = user?.role === 'superadmin';
    const [pageSize] = useState(25);

    const { data, isLoading, error, refetch } = useQuery<HistoryResponse>({
        queryKey: ['legacyReprocessHistory', pageSize],
        queryFn: () => fetchHistory(pageSize),
        refetchInterval: 10000, // Background polling
        enabled: isSuperAdmin,
    });

    const [isTagging, setIsTagging] = useState(false);
    const [isDialogOpen, setIsDialogOpen] = useState(false);
    const [isSuccessDialogOpen, setIsSuccessDialogOpen] = useState(false);

    const handleTagChunks = async () => {
        setIsDialogOpen(false); // Close confirmation dialog immediately
        setIsTagging(true);
        try {
            const response = await axiosInstance.post('/docs/maintenance/tag-legacy-chunks');
            // Show the success pop-up instead of just a toast
            setIsSuccessDialogOpen(true);
            toast.success("Migration process started.");
            // Migration is async, so we just notify the user. 
            // Stats will update as workers finish or after migration completes.
        } catch (err: any) {
            console.error("Failed to tag legacy chunks:", err);
            toast.error(err.response?.data?.msg || "Failed to start legacy chunk tagging.");
        } finally {
            setIsTagging(false);
        }
    };

    useEffect(() => {
        if (!isLoading && user && !isSuperAdmin) {
            router.push('/history');
        }
    }, [user, isSuperAdmin, router, isLoading]);

    if (!isSuperAdmin) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-4">
                <Loader2 className="h-10 w-10 animate-spin text-primary/60" />
                <p className="text-muted-foreground font-medium">Checking permissions...</p>
            </div>
        );
    }

    return (
        <div className="container mx-auto p-4 md:p-6 lg:p-8 space-y-6">
            <div className="flex items-center gap-4 mb-6">
                <Link href="/history">
                    <Button variant="outline" size="icon">
                        <ArrowLeft className="h-4 w-4" />
                    </Button>
                </Link>
                <div className="flex flex-col">
                    <h1 className="text-3xl font-bold tracking-tight">Legacy Reprocess History</h1>
                    <p className="text-muted-foreground">Detailed logs of all bulk reprocessing runs.</p>
                </div>

                <div className="ml-auto">
                    <AlertDialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
                        <AlertDialogTrigger asChild>
                            <Button
                                variant="destructive"
                                className="h-8 px-3 font-bold uppercase tracking-tight text-[10px] shadow-sm bg-gradient-to-r from-red-600 to-orange-600 hover:from-red-700 hover:to-orange-700 border-none"
                                disabled={isTagging}
                            >
                                {isTagging ? (
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                ) : (
                                    <RefreshCw className="mr-2 h-4 w-4" />
                                )}
                                Tag Legacy Chunks (Backfill)
                            </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent className="sm:max-w-md">
                            <AlertDialogHeader>
                                <AlertDialogTitle className="flex items-center gap-2">
                                    <AlertCircle className="h-5 w-5 text-red-500" />
                                    Are you sure?
                                </AlertDialogTitle>
                                <AlertDialogDescription className="text-sm">
                                    This will initiate a background process to scan all chunks and tag them with `extraction_source` field.
                                    <br /><br />
                                    This is a intensive one-time operation. Are you sure you want to proceed?
                                </AlertDialogDescription>
                            </AlertDialogHeader>
                            <AlertDialogFooter>
                                <AlertDialogCancel disabled={isTagging}>Cancel</AlertDialogCancel>
                                <AlertDialogAction
                                    onClick={(e) => {
                                        e.preventDefault();
                                        handleTagChunks();
                                    }}
                                    className="bg-red-600 hover:bg-red-700"
                                    disabled={isTagging}
                                >
                                    {isTagging ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                                    Yes, Start Tagging
                                </AlertDialogAction>
                            </AlertDialogFooter>
                        </AlertDialogContent>
                    </AlertDialog>

                    {/* Success Dialog */}
                    <AlertDialog open={isSuccessDialogOpen} onOpenChange={setIsSuccessDialogOpen}>
                        <AlertDialogContent className="sm:max-w-md">
                            <AlertDialogHeader>
                                <AlertDialogTitle className="flex items-center gap-2 text-green-600">
                                    <CheckCircle className="h-5 w-5" />
                                    Migration Started Successfully
                                </AlertDialogTitle>
                                <AlertDialogDescription className="text-sm">
                                    The background process has been initiated correctly.
                                    <br /><br />
                                    All legacy chunks will be tagged with the new `extraction_source` field. This may take some time to complete depending on the database size.
                                </AlertDialogDescription>
                            </AlertDialogHeader>
                            <AlertDialogFooter>
                                <AlertDialogAction className="bg-green-600 hover:bg-green-700">
                                    Got it, thanks!
                                </AlertDialogAction>
                            </AlertDialogFooter>
                        </AlertDialogContent>
                    </AlertDialog>
                </div>
            </div>

            <Card className="shadow-sm border-muted/40">
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-7">
                    <CardTitle className="text-xl font-semibold flex items-center gap-2">
                        <History className="h-5 w-5 text-muted-foreground" />
                        Run History
                    </CardTitle>
                    <div className="flex items-center gap-2">
                        <Button variant="outline" size="sm" onClick={() => refetch()} disabled={isLoading} className="h-8">
                            {isLoading ? <Loader2 className="h-3 w-3 animate-spin mr-2" /> : <RefreshCw className="h-3 w-3 mr-2" />}
                            Refresh
                        </Button>
                    </div>
                </CardHeader>
                <CardContent>
                    {isLoading && !data && (
                        <div className="flex flex-col items-center justify-center py-24 space-y-4">
                            <Loader2 className="h-10 w-10 animate-spin text-primary/60" />
                            <p className="text-muted-foreground font-medium animate-pulse">Loading history logs...</p>
                        </div>
                    )}

                    {error && (
                        <div className="flex flex-col items-center justify-center py-20 text-center">
                            <div className="bg-red-50 dark:bg-red-950/20 p-4 rounded-full mb-4">
                                <AlertCircle className="h-10 w-10 text-red-500" />
                            </div>
                            <h3 className="text-lg font-semibold text-red-900 dark:text-red-400">Failed to load history</h3>
                            <p className="text-muted-foreground max-w-md mt-2">{(error as Error).message}</p>
                            <Button variant="outline" onClick={() => refetch()} className="mt-6">Try Again</Button>
                        </div>
                    )}

                    {data && data.runs.length === 0 && (
                        <div className="flex flex-col items-center justify-center py-24 text-center opacity-60">
                            <History className="h-16 w-16 text-muted-foreground mb-4" />
                            <h3 className="text-xl font-medium">No history found</h3>
                            <p className="text-muted-foreground max-w-sm mt-2">History will appear here once you initiate the legacy reprocessing runner.</p>
                        </div>
                    )}

                    {data && data.runs.length > 0 && (
                        <div className="relative overflow-x-auto rounded-md border border-muted/30">
                            <Table>
                                <TableHeader className="bg-muted/30">
                                    <TableRow>
                                        <TableHead className="py-4 px-6 font-semibold">Started</TableHead>
                                        <TableHead className="font-semibold text-center">Trigger</TableHead>
                                        <TableHead className="font-semibold">Initiated By</TableHead>
                                        <TableHead className="font-semibold">Status</TableHead>
                                        <TableHead className="text-right font-semibold">Found</TableHead>
                                        <TableHead className="text-right font-semibold">Queued</TableHead>
                                        <TableHead className="text-right font-semibold">Skipped</TableHead>
                                        <TableHead className="font-semibold pl-8">Duration</TableHead>
                                    </TableRow>
                                </TableHeader>
                                <TableBody>
                                    {data.runs.map((run) => {
                                        const duration = run.end_time
                                            ? Math.round((new Date(run.end_time).getTime() - new Date(run.start_time).getTime()) / 1000)
                                            : null;

                                        return (
                                            <TableRow key={run.run_id} className="hover:bg-muted/10 transition-colors">
                                                <TableCell className="py-4 px-6 font-mono text-sm">
                                                    {format(new Date(run.start_time), 'MMM dd, yyyy HH:mm:ss')}
                                                </TableCell>
                                                <TableCell className="text-center">{getTriggerBadge(run.trigger_source)}</TableCell>
                                                <TableCell className="text-sm">
                                                    <div className="flex flex-col max-w-[180px]">
                                                        <span className="truncate font-medium" title={run.initiated_by}>{run.initiated_by}</span>
                                                        <span className="text-[10px] text-muted-foreground font-mono truncate">{run.run_id}</span>
                                                    </div>
                                                </TableCell>
                                                <TableCell>{getStatusBadge(run.status)}</TableCell>
                                                <TableCell className="text-right font-mono text-sm tabular-nums font-medium">{run.docs_found.toLocaleString()}</TableCell>
                                                <TableCell className="text-right font-mono text-sm tabular-nums text-primary/80">{run.docs_queued.toLocaleString()}</TableCell>
                                                <TableCell className="text-right font-mono text-sm tabular-nums text-muted-foreground">{run.docs_skipped.toLocaleString()}</TableCell>
                                                <TableCell className="text-sm pl-8">
                                                    {duration !== null ? (
                                                        <span className="font-medium">
                                                            {duration < 60
                                                                ? `${duration}s`
                                                                : `${Math.floor(duration / 60)}m ${duration % 60}s`}
                                                        </span>
                                                    ) : run.status === 'completed' ? (
                                                        <span className="text-muted-foreground italic">N/A</span>
                                                    ) : (
                                                        <div className="flex items-center text-amber-600 dark:text-amber-400 font-medium">
                                                            <Clock className="h-3 w-3 mr-2 animate-pulse" />
                                                            Running...
                                                        </div>
                                                    )}
                                                </TableCell>
                                            </TableRow>
                                        );
                                    })}
                                </TableBody>
                            </Table>
                        </div>
                    )}

                    {data && data.runs.length > 0 && (
                        <div className="flex items-center justify-between pt-6 px-1">
                            <p className="text-xs text-muted-foreground">
                                Showing latest 25 runs. Auto-refreshing every 10 seconds.
                            </p>
                            <div className="flex gap-2">
                                {/* Pagination buttons could go here if needed in future */}
                            </div>
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    );
}

export default withAuth(LegacyReprocessHistoryPage);
