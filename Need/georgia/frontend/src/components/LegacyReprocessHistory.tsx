"use client";

import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import axiosInstance from '@/lib/axiosInstance';
import { format } from 'date-fns';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { History, Loader2, User, Clock, CheckCircle, XCircle, AlertCircle } from 'lucide-react';

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

export default function LegacyReprocessHistory() {
  const [open, setOpen] = useState(false);
  
  const { data, isLoading, error, refetch } = useQuery<HistoryResponse>({
    queryKey: ['legacyReprocessHistory', 20],
    queryFn: () => fetchHistory(20),
    enabled: open,
    refetchInterval: open ? 10000 : false // Refetch every 10s when dialog is open
  });

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon" title="View Legacy Reprocess History" className="h-[42px] w-[42px]">
          <History className="h-5 w-5" />
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-7xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center justify-between">
            <span>Legacy Reprocess Run History</span>
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={() => refetch()}
              disabled={isLoading}
              className="text-xs"
            >
              {isLoading ? <Loader2 className="h-3 w-3 animate-spin mr-1" /> : null}
              Refresh
            </Button>
          </DialogTitle>
        </DialogHeader>
        
        {isLoading && (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            <span className="ml-3 text-muted-foreground">Loading history...</span>
          </div>
        )}
        
        {error && (
          <div className="text-red-600 py-6 px-4 bg-red-50 dark:bg-red-950/20 rounded-lg border border-red-200 dark:border-red-800">
            <p className="font-semibold">Error loading history</p>
            <p className="text-sm mt-1">{(error as Error).message}</p>
          </div>
        )}
        
        {data && data.runs.length === 0 && (
          <div className="text-center py-12">
            <History className="h-12 w-12 mx-auto text-muted-foreground/50 mb-3" />
            <p className="text-muted-foreground">No legacy reprocess runs found.</p>
            <p className="text-sm text-muted-foreground/70 mt-1">History will appear here after you trigger a reprocess.</p>
          </div>
        )}
        
        {data && data.runs.length > 0 && (
          <div className="rounded-lg border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[180px]">Started</TableHead>
                  <TableHead className="w-[120px]">Trigger</TableHead>
                  <TableHead className="w-[200px]">Initiated By</TableHead>
                  <TableHead className="w-[130px]">Status</TableHead>
                  <TableHead className="text-right w-[80px]">Found</TableHead>
                  <TableHead className="text-right w-[80px]">Queued</TableHead>
                  <TableHead className="text-right w-[80px]">Skipped</TableHead>
                  <TableHead className="w-[100px]">Duration</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.runs.map((run) => {
                  const duration = run.end_time 
                    ? Math.round((new Date(run.end_time).getTime() - new Date(run.start_time).getTime()) / 1000)
                    : null;
                  
                  return (
                    <TableRow key={run.run_id}>
                      <TableCell className="text-sm font-mono">
                        {format(new Date(run.start_time), 'MMM dd, HH:mm:ss')}
                      </TableCell>
                      <TableCell>{getTriggerBadge(run.trigger_source)}</TableCell>
                      <TableCell className="text-sm truncate max-w-[200px]" title={run.initiated_by}>
                        {run.initiated_by}
                      </TableCell>
                      <TableCell>{getStatusBadge(run.status)}</TableCell>
                      <TableCell className="text-right font-mono text-sm">{run.docs_found.toLocaleString()}</TableCell>
                      <TableCell className="text-right font-mono text-sm">{run.docs_queued.toLocaleString()}</TableCell>
                      <TableCell className="text-right font-mono text-sm">{run.docs_skipped.toLocaleString()}</TableCell>
                      <TableCell className="text-sm">
                        {duration !== null ? (
                          duration < 60 
                            ? `${duration}s` 
                            : `${Math.floor(duration / 60)}m ${duration % 60}s`
                        ) : run.status === 'completed' ? 'N/A' : (
                          <span className="text-yellow-600 dark:text-yellow-400 flex items-center">
                            <Clock className="h-3 w-3 mr-1 animate-pulse" />
                            Running...
                          </span>
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
          <div className="text-xs text-muted-foreground text-center pt-2">
            Showing {data.runs.length} most recent run{data.runs.length !== 1 ? 's' : ''}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
