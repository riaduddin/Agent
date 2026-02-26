"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useQuery, useMutation } from "@tanstack/react-query"; // Removed useQueryClient
import { isAxiosError } from "axios"; // Import isAxiosError only
import axiosInstance from "@/lib/axiosInstance"; // Assuming you have configured axios
import { useAuth } from "@/context/AuthContext"; // Import useAuth hook
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
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
} from "@/components/ui/alert-dialog"; // Import AlertDialog components
// Removed unused icons: FileClock, Hourglass. Re-added CheckCircle2. Removed Info, Database, SearchCode, Wrench, ServerCrash. Removed CheckCircle2, AlertTriangle
import {
  Loader2,
  XCircle,
  ExternalLink,
  PlayCircle,
  Trash2,
  RotateCcw,
} from "lucide-react"; // Added PlayCircle, Trash2, RotateCcw
import { format } from "date-fns"; // For formatting dates
import Link from "next/link"; // Import Link
import { toast } from "sonner"; // Import toast
import { cn } from "@/lib/utils";

interface VectorStatus {
  endpoint_connected: boolean;
  index_deployed: boolean;
  error: string | null;
}

// Removed unused type
// interface BulkProcessResult { ... }

// Interface for the stats fetched from Firestore
interface BulkStats {
  run_id: string;
  start_time: string | null; // ISO string
  end_time: string | null; // ISO string
  status: string; // e.g., 'scanning_gcs', 'tagging_and_enqueueing', 'enqueued', 'running', 'completed', 'error'
  gcs_prefix_used: string;
  files_found: number;
  files_tagged_for_split: number;
  chunks_split?: number;
  chunks_ocr_completed?: number;
  chunks_vectorized?: number;
  error_count?: number;
  errors: string[];
  completed: number;
  enqueued_count?: number;
}

// Interface for the new dashboard stats endpoint
interface DashboardStats {
  total_documents: number;
  total_chunks?: number; // Add optional total_chunks
  status_counts: { [key: string]: number };
}

// Interface for process pending response
interface ProcessPendingResponse {
  status: string;
  message: string;
  enqueued_count?: number;
  error_count?: number;
}

// Interface for clear data response
interface ClearDataResponse {
  status: string;
  message: string;
  details?: { [key: string]: { deleted: number } };
}

// Interface for reset stuck response
interface ResetStuckResponse {
  status: string;
  message: string;
  reset_count?: number;
}
// Removed unused CheckItem component definition

export default function InitialSetupStatisticsPage() {
  const { user } = useAuth(); // Use useAuth hook to get user
  const [vectorStatus, setVectorStatus] = useState<VectorStatus | null>(null);
  const [statusLoading, setStatusLoading] = useState<boolean>(true);
  const [statusError, setStatusError] = useState<string | null>(null);

  const [bulkProcessing, setBulkProcessing] = useState<boolean>(false);
  const [bulkInitiationResult, setBulkInitiationResult] = useState<{
    msg: string;
  } | null>(null);
  const [bulkError, setBulkError] = useState<string | null>(null);
  const [isConfirmDialogOpen, setIsConfirmDialogOpen] = useState(false);

  // States for Category Backfill
  const [isBackfillLoading, setIsBackfillLoading] = useState(false);

  const [bulkStats, setBulkStats] = useState<BulkStats | null>(null);
  const [statsLoading, setStatsLoading] = useState<boolean>(false); // Keep for bulk stats card
  // Remove manual dashboard loading state and error state, use useQuery's flags/error
  // const [dashboardStatsLoading, setDashboardStatsLoading] = useState<boolean>(false);
  // const [dashboardError, setDashboardError] = useState<string | null>(null); // Remove error state

  const {
    data: dashboardStats,
    isLoading: isDashboardLoading,
    isFetching: isDashboardFetching,
    error: dashboardQueryError,
    refetch: refetchDashboardStats,
  } = useQuery<DashboardStats, Error>({
    // Destructure isLoading, isFetching, error
    queryKey: ["dashboardStats"],
    queryFn: async () => {
      // Remove manual loading state management
      // setDashboardStatsLoading(true);
      // setDashboardError(null); // Remove manual error reset
      try {
        const response = await axiosInstance.get(
          "/system/processing-dashboard-stats"
        );
        return response.data;
      } catch (err: unknown) {
        // Use unknown type
        console.error("Error fetching dashboard stats:", err);
        // Removed unused errorMsg variable declaration and assignment
        // let errorMsg = "Failed to fetch dashboard stats";
        // if (isAxiosError(err) && err.response?.data?.message) {
        //     errorMsg = err.response.data.message;
        // }
        // setDashboardError(errorMsg); // Remove manual error set
        throw err; // Re-throw error for useQuery to handle
      }
      // Remove finally block setting loading state
    },
    enabled: true, // Enable the query to run automatically on mount
    refetchInterval: 3000, // Refetch every 3 seconds
    refetchOnWindowFocus: true, // Optional: Also refetch when window regains focus
  });

  const fetchVectorStatus = useCallback(async () => {
    setStatusLoading(true);
    setStatusError(null);
    setVectorStatus(null);
    try {
      const response = await axiosInstance.get<VectorStatus>(
        "/docs/status/vector-search"
      );
      setVectorStatus(response.data);
    } catch (error: unknown) {
      // Use unknown type
      console.error("Error fetching vector status:", error);
      let errorMsg = "Failed to fetch status";
      if (isAxiosError(error) && error.response?.data?.error) {
        errorMsg = error.response.data.error;
      } else if (error instanceof Error) {
        errorMsg = error.message;
      }
      setStatusError(errorMsg);
      setVectorStatus(null);
    } finally {
      setStatusLoading(false);
    }
  }, []);

  const fetchBulkStats = useCallback(async () => {
    setStatsLoading(true);
    try {
      const response = await axiosInstance.get<BulkStats>(
        "/system/bulk-process-stats"
      );
      setBulkStats(response.data);
      setBulkError(null);
    } catch (error: unknown) {
      // Use unknown type
      let is404 = false;
      if (isAxiosError(error) && error.response?.status === 404) {
        is404 = true;
      }

      if (is404) {
        setBulkStats(null);
        setBulkError(null);
      } else {
        console.error("Error fetching bulk stats:", error);
        let errorMsg = "Failed to fetch bulk process stats";
        if (isAxiosError(error) && error.response?.data?.message) {
          errorMsg = error.response.data.message;
        } else if (error instanceof Error) {
          errorMsg = error.message;
        }
        setBulkError(errorMsg);
        setBulkStats(null);
      }
    } finally {
      setStatsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchVectorStatus();
    fetchBulkStats();
    refetchDashboardStats();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const refreshAllStats = useCallback(() => {
    fetchBulkStats();
    refetchDashboardStats();
  }, [fetchBulkStats, refetchDashboardStats]);

  const executeBulkProcessInitiation = async () => {
    setBulkProcessing(true);
    setBulkError(null);
    setBulkInitiationResult(null);
    setBulkStats(null);
    try {
      const response = await axiosInstance.post<{ message: string }>(
        "/system/start-bulk-process",
        {
          gcs_prefix: "",
        }
      );
      setBulkInitiationResult({
        msg: response.data.message || "Bulk processing initiated.",
      });
      await refreshAllStats();
    } catch (error: unknown) {
      // Use unknown type
      console.error("Error initiating bulk processing:", error);
      let errorMsg = "Failed to initiate bulk processing";
      if (isAxiosError(error) && error.response?.data?.message) {
        errorMsg = error.response.data.message;
      } else if (error instanceof Error) {
        errorMsg = error.message;
      }
      setBulkError(errorMsg);
      setBulkInitiationResult(null);
    } finally {
      setBulkProcessing(false);
    }
  };

  const [processPendingLoading, setProcessPendingLoading] = useState(false);
  const processPendingMutation = useMutation<ProcessPendingResponse, Error>({
    mutationFn: () =>
      axiosInstance.post("/system/process-pending").then((res) => res.data),
    onMutate: () => {
      setProcessPendingLoading(true);
    },
    onSuccess: (data) => {
      toast.success("Processing Source Files Initiated", {
        description: data.message,
      });
      refreshAllStats();
    },
    onError: (err: unknown) => {
      // Use unknown type
      let errorMsg = "Failed to initiate processing for pending files.";
      if (isAxiosError(err) && err.response?.data?.message) {
        errorMsg = err.response.data.message;
      } else if (err instanceof Error) {
        errorMsg = err.message;
      }
      toast.error("Error Processing Source Files", {
        description: errorMsg,
      });
    },
    onSettled: () => {
      setProcessPendingLoading(false);
    },
  });

  const handleProcessPending = () => {
    processPendingMutation.mutate();
  };

  const [clearDataLoading, setClearDataLoading] = useState(false);
  const clearDataMutation = useMutation<ClearDataResponse, Error>({
    mutationFn: () =>
      axiosInstance
        .delete("/system/clear-processing-data")
        .then((res) => res.data),
    onMutate: () => {
      setClearDataLoading(true);
    },
    onSuccess: (data) => {
      toast.success("Processing Data Cleared", {
        description: data.message,
      });
      refreshAllStats();
    },
    onError: (err: unknown) => {
      // Use unknown type
      let errorMsg = "Failed to clear processing data.";
      if (isAxiosError(err) && err.response?.data?.message) {
        errorMsg = err.response.data.message;
      } else if (err instanceof Error) {
        errorMsg = err.message;
      }
      toast.error("Error Clearing Data", {
        description: errorMsg,
      });
    },
    onSettled: () => {
      setClearDataLoading(false);
    },
  });

  const handleClearData = () => {
    clearDataMutation.mutate();
  };

  // Mutation for resetting stuck documents
  const [resetStuckLoading, setResetStuckLoading] = useState(false);
  const resetStuckMutation = useMutation<ResetStuckResponse, Error>({
    mutationFn: () =>
      axiosInstance
        .post("/system/reset-stuck-documents")
        .then((res) => res.data),
    onMutate: () => {
      setResetStuckLoading(true);
    },
    onSuccess: (data) => {
      toast.success("Reset Stuck Documents Initiated", {
        description: data.message,
      });
      refreshAllStats(); // Refresh stats after resetting
    },
    onError: (err: unknown) => {
      let errorMsg = "Failed to reset stuck documents.";
      if (isAxiosError(err) && err.response?.data?.message) {
        errorMsg = err.response.data.message;
      } else if (err instanceof Error) {
        errorMsg = err.message;
      }
      toast.error("Error Resetting Stuck Documents", {
        description: errorMsg,
      });
    },
    onSettled: () => {
      setResetStuckLoading(false);
    },
  });

  const handleResetStuck = () => {
    resetStuckMutation.mutate();
  };

  // Mutation for starting the category backfill process
  const startBackfillMutation = useMutation({
    mutationFn: () => {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_API_URL;
      return axiosInstance.post(
        `${backendUrl}/backend/api/v2/system/start-categorization-batch`
      );
    },
    onMutate: () => {
      setIsBackfillLoading(true);
    },
    onSuccess: (response) => {
      toast.success("✅ Processing has been successfully initiated.", {
        description:
          response.data.msg ||
          "The categorization backfill process has started.",
      });
    },
    onError: (err: unknown) => {
      let errorMsg = "Failed to start categorization batch.";
      if (isAxiosError(err) && err.response?.data?.msg) {
        errorMsg = err.response.data.msg;
      }
      toast.error("Initiation Failed", {
        description: errorMsg,
      });
    },
    onSettled: () => {
      setIsBackfillLoading(false);
    },
  });

  const handleStartBackfill = () => {
    startBackfillMutation.mutate();
  };

  const getStatusVariant = (): "default" | "destructive" => {
    if (!vectorStatus) return "destructive";
    if (vectorStatus.endpoint_connected && vectorStatus.index_deployed)
      return "default";
    if (vectorStatus.endpoint_connected && !vectorStatus.index_deployed)
      return "destructive";
    return "destructive";
  };

  const getStatusTitle = (): string => {
    if (statusLoading) return "Checking Status...";
    if (statusError) return "Status Check Failed";
    if (!vectorStatus) return "Status Unavailable";
    if (vectorStatus.endpoint_connected && vectorStatus.index_deployed)
      return "Vector Search Ready";
    if (vectorStatus.endpoint_connected && !vectorStatus.index_deployed)
      return "Endpoint Connected, Index Not Deployed";
    return "Vector Search Not Connected";
  };

  const getStatusDescription = (): string => {
    if (statusLoading)
      return "Please wait while we check the connection to Vertex AI Vector Search.";
    if (statusError) return `Error: ${statusError}`;
    if (!vectorStatus) return "Could not retrieve status information.";
    if (vectorStatus.endpoint_connected && vectorStatus.index_deployed)
      return "Successfully connected to the Vector Search endpoint and the specified index is deployed.";
    if (vectorStatus.endpoint_connected && !vectorStatus.index_deployed)
      return `Connected to the endpoint, but the deployed index ID specified in the backend .env was not found. Ensure the index is deployed and the ID is correct. ${vectorStatus.error ? `Details: ${vectorStatus.error}` : ""
        }`;
    return `Failed to connect to the Vector Search endpoint. Check backend configuration (.env) and GCP setup. ${vectorStatus.error ? `Details: ${vectorStatus.error}` : ""
      }`;
  };

  return (
    <div className="flex-1 bg-[#F8F8F8] p-8 overflow-y-auto">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header Section */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
          <div>
            <h1
              className="text-[28px] font-bold text-[#255c5d] leading-tight"
              style={{ fontFamily: 'var(--unnamed-font-family-georgia)' }}
            >
              System Operations & Pulse
            </h1>
            <p
              className="text-xs text-[#707070] mt-1"
              style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
            >
              Monitor cloud infrastructure status, manage high-volume processing tasks, and audit platform-wide document health
            </p>
          </div>
          <Button
            onClick={refreshAllStats}
            disabled={statsLoading || isDashboardFetching}
            className="flex items-center gap-2 bg-[#255c5d] hover:bg-[#1E4E55] text-white font-bold uppercase text-[10px] tracking-widest rounded-[4px] h-11 px-8 transition-all shadow-md active:scale-95"
            style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
          >
            {statsLoading || isDashboardFetching ? <Loader2 className="h-3 w-3 animate-spin" /> : <RotateCcw className="h-3.5 w-3.5" />}
            Refresh Global Pulse
          </Button>
        </div>

        {/* Dashboard Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card className="border border-[#E0E0E0] rounded-[8px] overflow-hidden shadow-sm bg-white">
            <CardContent className="p-6">
              <p className="text-[10px] font-bold text-[#707070] uppercase tracking-widest mb-1">Total Repository</p>
              <div className="flex items-baseline gap-2">
                <h3 className="text-3xl font-bold text-[#255c5d]">{dashboardStats?.total_documents ?? 0}</h3>
                <span className="text-[10px] text-[#707070] font-medium">Documents</span>
              </div>
            </CardContent>
          </Card>
          <Card className="border border-[#E0E0E0] rounded-[8px] overflow-hidden shadow-sm bg-white">
            <CardContent className="p-6">
              <p className="text-[10px] font-bold text-[#707070] uppercase tracking-widest mb-1">Data Volume</p>
              <div className="flex items-baseline gap-2">
                <h3 className="text-3xl font-bold text-[#255c5d]">{dashboardStats?.total_chunks ?? 0}</h3>
                <span className="text-[10px] text-[#707070] font-medium">Atomic Chunks</span>
              </div>
            </CardContent>
          </Card>
          <Card className="border border-[#E0E0E0] rounded-[8px] overflow-hidden shadow-sm bg-white">
            <CardContent className="p-6">
              <p className="text-[10px] font-bold text-[#707070] uppercase tracking-widest mb-1">Source Health</p>
              <div className="flex items-baseline gap-2">
                <h3 className="text-3xl font-bold text-emerald-600">{dashboardStats?.status_counts?.Processed ?? 0}</h3>
                <span className="text-[10px] text-[#707070] font-medium">Indexed</span>
              </div>
            </CardContent>
          </Card>
          <Card className="border border-[#E0E0E0] rounded-[8px] overflow-hidden shadow-sm bg-white">
            <CardContent className="p-6">
              <p className="text-[10px] font-bold text-[#707070] uppercase tracking-widest mb-1">Pending Batch</p>
              <div className="flex items-baseline gap-2">
                <h3 className="text-3xl font-bold text-amber-500">{dashboardStats?.status_counts?.Pending ?? 0}</h3>
                <span className="text-[10px] text-[#707070] font-medium">Awaiting Action</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

          {/* Left Column: Process & Health */}
          <div className="lg:col-span-2 space-y-8">
            {/* Status Breakdown */}
            <Card className="border border-[#E0E0E0] rounded-[8px] overflow-hidden shadow-sm">
              <div className="bg-[#F1F8F8] px-6 py-4 border-b border-[#E0E0E0]">
                <h2 className="text-[11px] font-bold text-[#255c5d] uppercase tracking-[0.2em]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Processing Lifecycle Distribution</h2>
              </div>
              <CardContent className="p-8 bg-white">
                {isDashboardLoading ? (
                  <div className="flex flex-col items-center justify-center py-12 gap-4">
                    <Loader2 className="h-8 w-8 animate-spin text-[#255c5d]/20" />
                    <p className="text-xs text-[#707070] font-medium italic">Calculating distribution analytics...</p>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {dashboardStats && Object.entries(dashboardStats.status_counts).map(([status, count]) => (
                      <div key={status} className="space-y-2">
                        <div className="flex justify-between items-end">
                          <span className="text-[11px] font-bold text-[#255c5d] uppercase tracking-wider">{status.replace(/_/g, " ")}</span>
                          <span className="text-xs font-mono font-bold text-[#707070]">{count} Docs</span>
                        </div>
                        <div className="h-1.5 w-full bg-[#F0F0F0] rounded-full overflow-hidden">
                          <div
                            className={cn(
                              "h-full transition-all duration-1000",
                              status === 'Processed' ? "bg-emerald-500" : status === 'Pending' ? "bg-amber-400" : "bg-[#255c5d]"
                            )}
                            style={{ width: `${(count / (dashboardStats.total_documents || 1)) * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
              <CardFooter className="bg-[#F8F8F8] border-t border-[#E0E0E0] px-6 py-4 flex justify-between items-center">
                <p className="text-[10px] text-[#707070] font-medium italic">Snapshot based on latest repository audit</p>
                <Link href="/logs">
                  <Button variant="ghost" size="sm" className="text-[#255c5d] text-[10px] font-bold uppercase tracking-widest hover:bg-[#E8F3F3]">
                    View Event Logs
                    <ExternalLink className="h-3 w-3 ml-2" />
                  </Button>
                </Link>
              </CardFooter>
            </Card>

            {/* Maintenance Tools */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <Card className="border border-[#E0E0E0] rounded-[8px] overflow-hidden shadow-sm bg-white">
                <div className="p-6 space-y-4">
                  <h3 className="text-xs font-bold text-[#255c5d] uppercase tracking-wider mb-2">Batch Processing Management</h3>
                  <p className="text-[11px] text-[#707070] leading-relaxed font-medium">
                    Configure and monitor batch processing runs for GCS buckets.
                  </p>
                  <p className="text-[11px] text-[#707070] leading-relaxed font-medium">
                    Navigate to the batch processing page to manage configurations, start new runs, and view reports.
                  </p>
                  <Link href="/batch-processing" className="block">
                    <Button className="w-full bg-[#255c5d] hover:bg-[#1E4E55] text-white font-bold uppercase text-[10px] tracking-widest rounded-[4px] h-11 shadow-sm">
                      Go to Batch Processing
                    </Button>
                  </Link>
                </div>
              </Card>

              <Card className="border border-[#E0E0E0] rounded-[8px] overflow-hidden shadow-sm bg-white">
                <div className="p-6 space-y-4">
                  <h3 className="text-xs font-bold text-[#255c5d] uppercase tracking-wider mb-2">Service Backfills</h3>
                  <p className="text-[11px] text-[#707070] leading-relaxed font-medium">
                    Initialize a platform-wide metadata categorization sweep. Required for new access control policies.
                  </p>
                  <Link href="/category-backfill" className="block">
                    <Button className="w-full bg-[#255c5d] hover:bg-[#1E4E55] text-white font-bold uppercase text-[10px] tracking-widest rounded-[4px] h-11 shadow-sm">
                      Start Categorization Batch
                    </Button>
                  </Link>
                </div>
              </Card>

              <Card className="border border-[#E8F3F3] rounded-[8px] overflow-hidden shadow-sm bg-[#F1F8F8]/30">
                <div className="p-6 space-y-4">
                  <h3 className="text-xs font-bold text-[#255c5d] uppercase tracking-wider mb-2">Queue Recovery</h3>
                  <p className="text-[11px] text-[#707070] leading-relaxed font-medium">
                    Synchronize stuck cloud functions and reset documents lingering in intermediate processing states.
                  </p>
                  <Button
                    onClick={handleResetStuck}
                    disabled={resetStuckLoading}
                    variant="outline"
                    className="w-full border-amber-500 text-amber-600 hover:bg-amber-50 font-bold uppercase text-[10px] tracking-widest rounded-[4px] h-11"
                  >
                    {resetStuckLoading ? <Loader2 className="h-3 w-3 animate-spin mr-2" /> : <RotateCcw className="h-3 w-3 mr-2" />}
                    Reset Stuck Processes
                  </Button>
                </div>
              </Card>
            </div>
          </div>

          {/* Right Column: Infra & Controls */}
          <div className="space-y-8">
            {/* Infrastructure Status */}
            <Card className="border border-[#E0E0E0] rounded-[8px] overflow-hidden shadow-sm">
              <div className="bg-[#F1F8F8] px-6 py-4 border-b border-[#E0E0E0]">
                <h2 className="text-[11px] font-bold text-[#255c5d] uppercase tracking-[0.2em]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Cloud Registry Status</h2>
              </div>
              <CardContent className="p-6 bg-white space-y-6">
                <div className="flex items-center justify-between">
                  <div className="flex flex-col">
                    <span className="text-[10px] font-bold text-[#255c5d] uppercase tracking-widest">Vector Endpoint</span>
                    <span className="text-[11px] font-medium text-[#707070]">
                      {statusLoading ? "Validating..." : vectorStatus?.endpoint_connected ? "Operational" : "Disconnected"}
                    </span>
                  </div>
                  <div className={cn(
                    "w-2.5 h-2.5 rounded-full shadow-[0_0_8px_rgba(0,0,0,0.1)]",
                    statusLoading ? "bg-gray-300" : vectorStatus?.endpoint_connected ? "bg-emerald-500" : "bg-red-500"
                  )} />
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex flex-col">
                    <span className="text-[10px] font-bold text-[#255c5d] uppercase tracking-widest">Index Deployment</span>
                    <span className="text-[11px] font-medium text-[#707070]">
                      {statusLoading ? "Scanning..." : vectorStatus?.index_deployed ? "Active Index" : "No Active Index"}
                    </span>
                  </div>
                  <div className={cn(
                    "w-2.5 h-2.5 rounded-full shadow-[0_0_8px_rgba(0,0,0,0.1)]",
                    statusLoading ? "bg-gray-300" : vectorStatus?.index_deployed ? "bg-emerald-500" : "bg-red-500"
                  )} />
                </div>

                {!statusLoading && statusError && (
                  <div className="p-3 bg-red-50 border border-red-100 rounded text-[10px] text-red-700 font-medium leading-relaxed">
                    {statusError}
                  </div>
                )}

                <Button
                  onClick={fetchVectorStatus}
                  disabled={statusLoading}
                  variant="ghost"
                  className="w-full text-[#255c5d] text-[10px] font-bold uppercase tracking-widest hover:bg-[#E8F3F3] h-10 border border-[#E0E0E0]"
                >
                  Re-validate Infrastructure
                </Button>
              </CardContent>
            </Card>

            {/* Run Initiation */}
            <Card className="border border-[#255c5d] rounded-[8px] overflow-hidden shadow-lg bg-[#255c5d]">
              <CardContent className="p-8 space-y-6">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-white/10 rounded-full">
                    <PlayCircle className="h-5 w-5 text-white" />
                  </div>
                  <h3 className="text-white text-lg font-bold" style={{ fontFamily: 'var(--unnamed-font-family-georgia)' }}>Initiate Bulk Run</h3>
                </div>
                <p className="text-white/80 text-[11px] font-medium leading-relaxed">
                  Recursively scan the entire GCS repository for new or modified documentation to harmonize with the vector store.
                </p>

                <AlertDialog open={isConfirmDialogOpen} onOpenChange={setIsConfirmDialogOpen}>
                  <AlertDialogTrigger asChild>
                    <Button
                      disabled={bulkProcessing || statusLoading || !vectorStatus?.index_deployed}
                      className="w-full bg-white text-[#255c5d] hover:bg-white/90 font-bold uppercase text-[10px] tracking-widest rounded-[4px] h-12 shadow-md active:scale-95 transition-all"
                    >
                      {bulkProcessing ? <Loader2 className="h-4 w-4 animate-spin" /> : "Start Global Harvest"}
                    </Button>
                  </AlertDialogTrigger>
                  <AlertDialogContent className="rounded-[12px] border border-[#E0E0E0] p-8">
                    <AlertDialogHeader>
                      <AlertDialogTitle className="text-[24px] font-bold text-[#255c5d]" style={{ fontFamily: 'var(--unnamed-font-family-georgia)' }}>Confirm System Harvest</AlertDialogTitle>
                      <AlertDialogDescription className="text-sm text-[#707070] leading-relaxed pt-2" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>
                        You are about to initiate a global scan of the GCS repository. This operation will cross-reference existing documentation and queue missing entries for parallel Document AI processing. This may incur significant compute costs.
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter className="pt-6">
                      <AlertDialogCancel className="rounded-[4px] border border-[#E0E0E0] text-[10px] font-bold uppercase tracking-widest h-11 px-8">Abort</AlertDialogCancel>
                      <AlertDialogAction
                        onClick={executeBulkProcessInitiation}
                        className="rounded-[4px] bg-[#255c5d] hover:bg-[#1E4E55] text-white text-[10px] font-bold uppercase tracking-widest h-11 px-8"
                      >
                        Execute Run
                      </AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>

                <div className="space-y-3 pt-4 border-t border-white/10">
                  <div className="flex items-center gap-3">
                    <p className="text-[10px] text-white/60 font-medium italic">Latest Harvest:</p>
                    <p className="text-[10px] text-white font-bold">{bulkStats?.run_id ? `#${bulkStats.run_id.slice(0, 8)}` : "None"}</p>
                  </div>
                  <div className="flex items-center gap-3">
                    <p className="text-[10px] text-white/60 font-medium italic">Outcome:</p>
                    <p className={cn(
                      "text-[10px] font-bold uppercase",
                      bulkStats?.status === 'completed' ? "text-emerald-300" : "text-amber-300"
                    )}>
                      {bulkStats?.status ?? "Inactive"}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Data Purge - Use Caution */}
            <Card className="border border-red-100 rounded-[8px] overflow-hidden shadow-sm bg-red-50/20">
              <div className="p-6 space-y-4">
                <h3 className="text-xs font-bold text-red-700 uppercase tracking-wider mb-1 flex items-center gap-2">
                  <Trash2 className="h-3 w-3" />
                  Platform Reset
                </h3>
                <p className="text-[10px] text-[#707070] font-medium leading-relaxed">
                  Wipe all cached processing metadata and reset internal tracking IDs. This does <strong>not</strong> delete physical files.
                </p>

                <AlertDialog>
                  <AlertDialogTrigger asChild>
                    <Button variant="outline" className="w-full border-red-200 text-red-600 hover:bg-red-50 font-bold uppercase text-[10px] tracking-widest rounded-[4px] h-10 transition-colors shadow-sm">
                      Clear All Platform Metadata
                    </Button>
                  </AlertDialogTrigger>
                  <AlertDialogContent className="rounded-[12px] border border-red-200 p-8 shadow-2xl">
                    <AlertDialogHeader>
                      <AlertDialogTitle className="text-[24px] font-bold text-red-700" style={{ fontFamily: 'var(--unnamed-font-family-georgia)' }}>Nuclear Action Detected</AlertDialogTitle>
                      <AlertDialogDescription className="text-sm text-[#707070] leading-relaxed pt-2" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>
                        Warning: This operation will permanently delete all document processing history and metadata from Firestore. The platform will lose knowledge of previously indexed documents. Physical storage in GCS will remain intact.
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter className="pt-6">
                      <AlertDialogCancel className="rounded-[4px] border border-[#E0E0E0] text-[10px] font-bold uppercase tracking-widest h-11 px-8">Abort</AlertDialogCancel>
                      <AlertDialogAction
                        onClick={handleClearData}
                        className="rounded-[4px] bg-red-600 hover:bg-red-700 text-white text-[10px] font-bold uppercase tracking-widest h-11 px-8"
                      >
                        Reset Metadata Repository
                      </AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
