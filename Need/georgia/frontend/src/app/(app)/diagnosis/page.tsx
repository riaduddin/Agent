/* eslint-disable @typescript-eslint/no-unused-vars */
"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"; // Added useQuery, useMutation, useQueryClient
import { isAxiosError } from "axios"; // Import axios and isAxiosError
import axiosInstance from "@/lib/axiosInstance";
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
import {
  Loader2,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Info,
  Database,
  SearchCode,
  ExternalLink,
  Wrench,
  ServerCrash,
  Settings2,
} from "lucide-react"; // Added Wrench, ServerCrash, Settings2
import { toast } from "sonner"; // Added sonner
import { ProcessorRulesCard } from "@/components/diagnosis/ProcessorRulesCard"; // Import the new card

import { cn } from "@/lib/utils";

// --- Interfaces ---
interface CheckResult {
  status: "OK" | "Error" | "Warning" | "Info";
  detail: string;
  value?: string;
  needs_manual_creation?: boolean; // Added flag for Firestore check
}

interface VectorSearchResult extends CheckResult {
  endpoint_connected?: boolean;
  index_deployed?: boolean;
  setup_needed?: boolean; // Added from backend check
}

interface EnvVarResults {
  [key: string]: CheckResult;
}

interface DiagnosisResults {
  "Configuration Variables": EnvVarResults;
  "Google Cloud Storage": CheckResult;
  Firestore: CheckResult;
  "Document AI": CheckResult;
  "Vertex AI Vector Search": VectorSearchResult;
  "Vertex AI Gemini": CheckResult;
}

interface IndexCreationResult {
  index: string;
  status: "Initiated" | "Exists" | "Error" | "Skipped";
  detail: string;
}

interface IndexCreationResponse {
  status: string;
  message: string;
  results?: IndexCreationResult[];
  error?: string; // Added for top-level errors
}

interface VectorSetupResult {
  endpoint_status: string;
  endpoint_id: string | null;
  endpoint_name: string | null;
  index_status: string;
  index_id: string | null;
  index_name: string | null;
  deployment_status: string;
  deployed_index_id: string | null;
  error: string | null;
  env_vars_to_set?: { [key: string]: string };
  message?: string;
}

interface DbSetupResult {
  status: "Exists" | "Initiated" | "Error" | "Info";
  message: string;
  error_details?: string;
}

// Define a union type for the result prop
type DiagnosisCheckResult = CheckResult | VectorSearchResult;

// --- Helper Components ---
const CheckItem: React.FC<{
  title: string;
  result: DiagnosisCheckResult | null;
  loading: boolean;
}> = ({ title, result, loading }) => {
  let Icon = Loader2;
  let iconColor = "text-[#A0A0A0]";
  let statusText = "Checking...";
  let details = "";
  let badgeColor = "bg-gray-50 text-[#707070] border-gray-200";

  if (loading || !result) {
    Icon = Loader2;
    iconColor = "text-[#255c5d] animate-spin";
  } else {
    const vectorResult = result as VectorSearchResult;

    if (title === "Vertex AI Vector Search" || title === "Vector Search Engine") {
      if (vectorResult.setup_needed) {
        Icon = AlertTriangle;
        iconColor = "text-yellow-600";
        statusText = "Setup Needed";
        badgeColor = "bg-yellow-50 text-yellow-700 border-yellow-200";
        details = vectorResult.detail || "Configuration missing. Use setup button.";
      } else if (vectorResult.endpoint_connected && vectorResult.index_deployed) {
        Icon = CheckCircle2;
        iconColor = "text-[#255c5d]";
        statusText = "Ready";
        badgeColor = "bg-[#E8F3F3] text-[#255c5d] border-[#255c5d]/20";
        details = vectorResult.detail || "Endpoint connected and index deployed.";
      } else {
        Icon = XCircle;
        iconColor = "text-red-600";
        statusText = "Restricted";
        badgeColor = "bg-red-50 text-red-700 border-red-200";
        details = vectorResult.detail || "Connection failure detected.";
      }
    } else if (result.status === "OK") {
      Icon = CheckCircle2;
      iconColor = "text-[#255c5d]";
      statusText = "Operational";
      badgeColor = "bg-[#E8F3F3] text-[#255c5d] border-[#255c5d]/20";
      details = result.value ? `Connected (${result.value})` : "Verified";
    } else {
      Icon = XCircle;
      iconColor = "text-red-600";
      statusText = "Error";
      badgeColor = "bg-red-50 text-red-700 border-red-200";
      details = result.detail || "Service unreachable.";
    }
  }

  return (
    <div className="flex items-center justify-between py-4 border-b border-[#E0E0E0] last:border-0 hover:bg-gray-50 transition-colors px-2">
      <div className="flex items-center gap-4">
        <div className={cn("p-2 rounded-full", loading ? "bg-transparent" : "bg-white border border-[#E0E0E0]")}>
          <Icon className={cn("h-4 w-4", iconColor)} />
        </div>
        <div>
          <p className="text-xs font-bold text-[#255c5d] uppercase tracking-wider" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>{title}</p>
          <p className="text-[11px] text-[#707070] font-medium leading-normal max-w-md">{details || statusText}</p>
        </div>
      </div>
      <div className={cn("px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-widest border", badgeColor)}>
        {statusText}
      </div>
    </div>
  );
};

export default function DiagnosisPage() {
  const { user } = useAuth();
  const [error, setError] = useState<string | null>(null);
  const [indexLoading, setIndexLoading] = useState<boolean>(false);
  const [indexResponse, setIndexResponse] = useState<IndexCreationResponse | null>(null);
  const [vectorSetupLoading, setVectorSetupLoading] = useState<boolean>(false);
  const [vectorSetupResponse, setVectorSetupResponse] = useState<VectorSetupResult | null>(null);
  const [dbSetupLoading, setDbSetupLoading] = useState<boolean>(false);
  const [dbSetupResponse, setDbSetupResponse] = useState<DbSetupResult | null>(null);
  const queryClient = useQueryClient();

  const {
    data: results,
    isLoading: loading,
    error: queryError,
    refetch: runDiagnosisCheck,
  } = useQuery<DiagnosisResults>({
    queryKey: ["diagnosis"],
    queryFn: async () => {
      const response = await axiosInstance.get("/system/diagnosis");
      return response.data;
    },
    refetchOnWindowFocus: false,
  });

  useEffect(() => {
    if (queryError) {
      let message = "Failed to run diagnosis check";
      if (isAxiosError(queryError) && queryError.response?.data?.msg) {
        message = queryError.response.data.msg;
      } else if (queryError instanceof Error) {
        message = queryError.message;
      }
      setError(message);
    } else {
      setError(null);
    }
  }, [queryError]);

  const createIndexMutation = useMutation<IndexCreationResponse, Error>({
    mutationFn: () => axiosInstance.post("/system/create-indexes").then((res) => res.data),
    onMutate: () => {
      setIndexLoading(true);
      setIndexResponse(null);
    },
    onSuccess: (data) => {
      setIndexResponse(data);
      toast.success("Firestore index creation initiated.");
    },
    onError: (err: unknown) => {
      let errorMsg = "Index creation failed.";
      if (isAxiosError(err)) errorMsg = err.response?.data?.message || err.message;
      toast.error(errorMsg);
    },
    onSettled: () => setIndexLoading(false),
  });

  const setupVectorSearchMutation = useMutation<VectorSetupResult, Error>({
    mutationFn: () => axiosInstance.post("/system/setup-vector-search").then((res) => res.data),
    onMutate: () => {
      setVectorSetupLoading(true);
      setVectorSetupResponse(null);
    },
    onSuccess: (data) => {
      setVectorSetupResponse(data);
      toast.success("Vector Search setup completed.");
      setTimeout(() => queryClient.invalidateQueries({ queryKey: ["diagnosis"] }), 2000);
    },
    onSettled: () => setVectorSetupLoading(false),
  });

  const setupDbMutation = useMutation<DbSetupResult, Error, { locationId: string }>({
    mutationFn: (variables) => axiosInstance.post("/system/setup-firestore-database", variables).then((res) => res.data),
    onMutate: () => {
      setDbSetupLoading(true);
      setDbSetupResponse(null);
    },
    onSuccess: (data) => {
      setDbSetupResponse(data);
      toast.success("Database check finished.");
      setTimeout(() => queryClient.invalidateQueries({ queryKey: ["diagnosis"] }), 2000);
    },
    onSettled: () => setDbSetupLoading(false),
  });

  const handleCreateIndex = useCallback(() => createIndexMutation.mutate(), [createIndexMutation]);
  const handleSetupVectorSearch = useCallback(() => setupVectorSearchMutation.mutate(), [setupVectorSearchMutation]);
  const handleSetupDatabase = useCallback(() => setupDbMutation.mutate({ locationId: "us-central1" }), [setupDbMutation]);

  if (user?.role === "user") {
    return (
      <div className="flex-1 flex items-center justify-center bg-[#F8F8F8] p-8">
        <div className="bg-white p-12 rounded-[12px] border border-[#E0E0E0] shadow-xl text-center max-w-md">
          <ServerCrash className="h-16 w-16 text-red-500 mx-auto mb-6" />
          <h2 className="text-[24px] font-bold text-[#255c5d] mb-4" style={{ fontFamily: 'var(--unnamed-font-family-georgia)' }}>Access Restricted</h2>
          <p className="text-sm text-[#707070] leading-relaxed" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>
            System diagnosis and infrastructure management are reserved for platform administrators. Please contact support if you require clearance.
          </p>
        </div>
      </div>
    );
  }

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
              System Management & Infrastructure
            </h1>
            <p
              className="text-xs text-[#707070] mt-1"
              style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
            >
              Audit core system connectivity, manage cloud resources, and reconcile platform environment variables
            </p>
          </div>
          <Button
            onClick={() => runDiagnosisCheck()}
            disabled={loading}
            className="flex items-center gap-2 bg-[#255c5d] hover:bg-[#1E4E55] text-white font-bold uppercase text-[10px] tracking-widest rounded-[4px] h-11 px-8 transition-all shadow-md active:scale-95"
            style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
          >
            {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : <Settings2 className="h-3 w-3" />}
            Refresh System Pulse
          </Button>
        </div>

        {error && (
          <Alert variant="destructive" className="bg-red-50 border-red-200 text-red-800 rounded-[8px]">
            <XCircle className="h-4 w-4" />
            <AlertTitle className="font-bold uppercase text-[10px] tracking-widest mb-1">Critical Diagnostic Failure</AlertTitle>
            <AlertDescription className="text-sm font-medium">{error}</AlertDescription>
          </Alert>
        )}

        {/* Diagnostic Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

          {/* Main Service Status */}
          <div className="lg:col-span-2 space-y-8">
            <Card className="border border-[#E0E0E0] rounded-[8px] overflow-hidden shadow-sm">
              <div className="bg-[#F1F8F8] px-6 py-4 border-b border-[#E0E0E0] flex justify-between items-center">
                <h2 className="text-[11px] font-bold text-[#255c5d] uppercase tracking-[0.2em]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Core Service Health</h2>
                <span className="text-[10px] text-[#255c5d]/60 font-medium italic">Validated via Cloud SDK</span>
              </div>
              <CardContent className="p-0 bg-white">
                <div className="flex flex-col">
                  {results ? (
                    <>
                      <CheckItem title="Google Cloud Storage" result={results?.["Google Cloud Storage"]} loading={loading} />
                      <CheckItem title="Vector Search Engine" result={results?.["Vertex AI Vector Search"]} loading={loading} />
                      <CheckItem title="Gemini 1.5 Pro" result={results?.["Vertex AI Gemini"]} loading={loading} />
                      <CheckItem title="Document Intelligence" result={results?.["Document AI"]} loading={loading} />
                      <CheckItem title="Managed Firestore" result={results?.Firestore} loading={loading} />
                    </>
                  ) : (
                    <div className="py-20 text-center space-y-4">
                      <Loader2 className="h-8 w-8 animate-spin text-[#255c5d]/20 mx-auto" />
                      <p className="text-xs text-[#707070] font-medium italic">Initializing system scan...</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Processor Rules / External Logic */}
            <ProcessorRulesCard />
          </div>

          {/* Environment & Metadata */}
          <div className="space-y-8">
            <Card className="border border-[#E0E0E0] rounded-[8px] overflow-hidden shadow-sm">
              <div className="bg-[#F1F8F8] px-6 py-4 border-b border-[#E0E0E0]">
                <h2 className="text-[11px] font-bold text-[#255c5d] uppercase tracking-[0.2em]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Environment Audit</h2>
              </div>
              <CardContent className="p-4 bg-white">
                <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
                  {results ? Object.entries(results["Configuration Variables"]).map(([key, result]) => (
                    <div key={key} className="flex flex-col gap-1.5 p-3 rounded-md bg-[#F8F8F8] border border-[#E0E0E0]">
                      <span className="text-[10px] font-mono font-bold text-[#255c5d] break-all">{key}</span>
                      <div className="flex items-center gap-2">
                        <span className={cn(
                          "w-2 h-2 rounded-full",
                          result.status === 'OK' ? "bg-emerald-500" : "bg-red-500"
                        )} />
                        <span className="text-[11px] text-[#707070] truncate">{result.value || 'NULL'}</span>
                      </div>
                    </div>
                  )) : (
                    <div className="py-10 text-center italic text-[#A0A0A0] text-xs">Waiting for environment data...</div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Infrastructure Actions */}
            <Card className="border border-[#E8F3F3] rounded-[8px] overflow-hidden shadow-md bg-[#F1F8F8]/30">
              <div className="px-6 py-5 space-y-4">
                <h3 className="text-xs font-bold text-[#255c5d] uppercase tracking-wider mb-2" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Administrative Tools</h3>

                <div className="space-y-3">
                  <Button
                    onClick={handleSetupDatabase}
                    disabled={loading || dbSetupLoading}
                    variant="outline"
                    className="w-full border-[#255c5d] text-[#255c5d] font-bold uppercase text-[10px] tracking-widest rounded-[4px] h-11 hover:bg-[#E8F3F3] justify-start px-4 transition-colors"
                  >
                    <Wrench className="h-3.5 w-3.5 mr-3" />
                    Verify Database Schema
                  </Button>

                  <Button
                    onClick={handleSetupVectorSearch}
                    disabled={loading || vectorSetupLoading}
                    variant="outline"
                    className="w-full border-[#255c5d] text-[#255c5d] font-bold uppercase text-[10px] tracking-widest rounded-[4px] h-11 hover:bg-[#E8F3F3] justify-start px-4 transition-colors"
                  >
                    <Database className="h-3.5 w-3.5 mr-3" />
                    Reconcile Vector Index
                  </Button>

                  <Button
                    onClick={handleCreateIndex}
                    disabled={loading || indexLoading}
                    variant="outline"
                    className="w-full border-[#255c5d] text-[#255c5d] font-bold uppercase text-[10px] tracking-widest rounded-[4px] h-11 hover:bg-[#E8F3F3] justify-start px-4 transition-colors"
                  >
                    <SearchCode className="h-3.5 w-3.5 mr-3" />
                    Sync Search Indexes
                  </Button>
                </div>

                <p className="text-[10px] text-[#255c5d]/60 italic font-medium pt-2">
                  * Operations above interact directly with GCP Infrastructure. Use with extreme caution.
                </p>
              </div>
            </Card>
          </div>
        </div>

        {/* Detailed Alerts for Setup */}
        {(dbSetupResponse || vectorSetupResponse?.env_vars_to_set) && (
          <div className="space-y-4 pt-4">
            {dbSetupResponse && (
              <Alert className={cn(
                "rounded-[8px] border",
                dbSetupResponse.status === "Error" ? "bg-red-50 border-red-200 text-red-800" : "bg-emerald-50 border-emerald-200 text-emerald-800"
              )}>
                <Info className="h-4 w-4" />
                <AlertTitle className="font-bold uppercase text-[10px] tracking-widest">Firestore Database Reconcilliation</AlertTitle>
                <AlertDescription className="text-sm">{dbSetupResponse.message}</AlertDescription>
              </Alert>
            )}
            {vectorSetupResponse?.env_vars_to_set && (
              <Alert className="bg-amber-50 border-amber-200 text-amber-900 rounded-[8px]">
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle className="font-bold uppercase text-[10px] tracking-widest">Infrastructure Update Required</AlertTitle>
                <AlertDescription className="space-y-3">
                  <p className="text-sm font-medium">New cloud resources detected. Synchronize your backend environment with the following parameters:</p>
                  <pre className="p-4 bg-white/50 border border-amber-200 rounded text-xs font-mono font-bold">
                    {Object.entries(vectorSetupResponse.env_vars_to_set).map(([k, v]) => `${k}=${v}`).join("\n")}
                  </pre>
                </AlertDescription>
              </Alert>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
