/* eslint-disable @typescript-eslint/no-empty-object-type */
/* eslint-disable @typescript-eslint/no-unused-vars */
"use client";

import React, { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import axiosInstance from "@/lib/axiosInstance";
import { useAuth } from "@/context/AuthContext";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "sonner";
import { Loader2, RefreshCw, FolderX, Trash2 } from "lucide-react";
import { Input } from "@/components/ui/input"; // Import Input component
import { Label } from "@/components/ui/label"; // Import Label component
import { cn } from "@/lib/utils";

// TODO: Define interfaces for API responses
interface BucketListResponse extends Array<string> { }
interface ConfigResponse {
  default_bucket_name: string | null;
}
interface ProcessedFile {
  file_gcs_path: string;
  file_size: number;
  status: string;
  first_seen_timestamp: { _seconds: number; _nanoseconds: number };
}
interface RunReport {
  run_id: string;
  bucket_name: string;
  start_timestamp: { _seconds: number; _nanoseconds: number };
  end_timestamp?: { _seconds: number; _nanoseconds: number };
  status: string;
  total_files_scanned: number;
  new_files_found: number;
  previously_processed_files: number;
  error_message?: string;
}
interface IgnoredFoldersResponse {
  ignored_folders: string[];
}

export default function BatchProcessingPage() {
  const { user } = useAuth();
  const [selectedBucket, setSelectedBucket] = useState<string>("");
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [newIgnoredFolder, setNewIgnoredFolder] = useState<string>(""); // State for new ignored folder input

  const {
    data: buckets,
    isLoading: isLoadingBuckets,
    error: bucketsError,
    refetch: refetchBuckets,
  } = useQuery<BucketListResponse, Error>({
    queryKey: ["gcsBuckets"],
    queryFn: async () => {
      const response = await axiosInstance.get("/batch/buckets");
      return response.data;
    },
    enabled: user?.role === "admin" || user?.role === "superadmin",
  });

  // Fetch the current default bucket config
  const {
    data: config,
    isLoading: isLoadingConfig,
    refetch: refetchConfig,
  } = useQuery<ConfigResponse, Error>({
    queryKey: ["batchConfig"],
    queryFn: async () => {
      const response = await axiosInstance.get("/batch/config");
      return response.data;
    },
    enabled: user?.role === "admin" || user?.role === "superadmin",
  });

  useEffect(() => {
    if (config?.default_bucket_name) {
      setSelectedBucket(config.default_bucket_name);
    }
  }, [config]);

  // Mutation to save the default bucket
  const saveConfigMutation = useMutation({
    mutationFn: (bucketName: string) => {
      return axiosInstance.post("/batch/config", { bucket_name: bucketName });
    },
    onSuccess: () => {
      toast.success("Default bucket saved successfully.");
    },
    onError: (error) => {
      toast.error("Failed to save default bucket.");
    },
  });

  const handleSaveConfig = () => {
    if (selectedBucket) {
      saveConfigMutation.mutate(selectedBucket);
    } else {
      toast.warning("Please select a bucket first.");
    }
  };

  // Mutation to start a new run
  const startRunMutation = useMutation({
    mutationFn: (bucketName: string) => {
      return axiosInstance.post("/batch/start", { bucket_name: bucketName });
    },
    onSuccess: (response) => {
      toast.success("Batch processing run started.", {
        description: response.data.message,
      });
      refetchReports();
    },
    onError: (error) => {
      toast.error("Failed to start batch processing run.");
    },
  });

  const handleStartRun = () => {
    if (selectedBucket) {
      startRunMutation.mutate(selectedBucket);
    } else {
      toast.warning("Please select a bucket to process.");
    }
  };

  const handleRefreshAll = () => {
    refetchBuckets();
    refetchConfig();
    refetchReports();
    toast.info("Refreshing all data...");
  };

  // Fetch the list of runs
  const {
    data: reports,
    isLoading: isLoadingReports,
    refetch: refetchReports,
  } = useQuery<RunReport[], Error>({
    queryKey: ["batchReports"],
    queryFn: async () => {
      const response = await axiosInstance.get("/batch/reports");
      return response.data;
    },
    enabled: user?.role === "admin" || user?.role === "superadmin",
    refetchInterval: 5000, // Refetch every 5 seconds to get status updates
  });

  useEffect(() => {
    if (reports && reports.length > 0 && !selectedRunId) {
      setSelectedRunId(reports[0].run_id);
    }
  }, [reports, selectedRunId]);

  const selectedReport = reports?.find(
    (report) => report.run_id === selectedRunId
  );

  // State for file list pagination
  const [filesCursor, setFilesCursor] = useState<string | null>(null);

  // Fetch the list of files for the selected run
  const { data: processedFiles, isLoading: isLoadingFiles } = useQuery<
    { files: ProcessedFile[]; next_cursor: string | null },
    Error
  >({
    queryKey: ["processedFiles", selectedRunId, filesCursor],
    queryFn: async () => {
      const response = await axiosInstance.get(
        `/batch/reports/${selectedRunId}/files`,
        {
          params: {
            limit: 10, // 10 files per page
            start_after: filesCursor,
          },
        }
      );
      return response.data;
    },
    enabled:
      (!!selectedRunId && user?.role === "admin") ||
      user?.role === "superadmin",
  });

  // Fetch ignored folders
  const {
    data: ignoredFoldersData,
    isLoading: isLoadingIgnoredFolders,
    refetch: refetchIgnoredFolders,
  } = useQuery<IgnoredFoldersResponse, Error>({
    queryKey: ["ignoredFolders"],
    queryFn: async () => {
      const response = await axiosInstance.get("/batch/ignored-folders");
      return response.data;
    },
    enabled: user?.role === "admin" || user?.role === "superadmin",
    refetchInterval: 10000, // Refetch every 10 seconds
  });

  // Mutation to add an ignored folder
  const addIgnoredFolderMutation = useMutation({
    mutationFn: (folderName: string) => {
      return axiosInstance.post("/batch/ignored-folders", {
        folder_name: folderName,
      });
    },
    onSuccess: () => {
      toast.success("Folder added to ignored list.");
      setNewIgnoredFolder(""); // Clear input
      refetchIgnoredFolders(); // Refresh the list
    },
    onError: (error) => {
      toast.error("Failed to add folder to ignored list.");
    },
  });

  const handleAddIgnoredFolder = () => {
    if (newIgnoredFolder.trim()) {
      addIgnoredFolderMutation.mutate(newIgnoredFolder.trim());
    } else {
      toast.warning("Please enter a folder name.");
    }
  };

  // Mutation to delete an ignored folder
  const deleteIgnoredFolderMutation = useMutation({
    mutationFn: (folderPath: string) => {
      return axiosInstance.delete("/batch/ignored-folders", {
        data: { folder_path: folderPath },
      });
    },
    onSuccess: () => {
      toast.success("Folder removed from ignored list.");
      refetchIgnoredFolders(); // Refresh the list
    },
    onError: (error) => {
      toast.error("Failed to remove folder from ignored list.");
    },
  });

  const allowedRoles = ["admin", "superadmin"];

  if (!allowedRoles.includes(user?.role || "")) {
    return (
      <div className="p-4 md:p-6 lg:p-8">
        <p>You do not have permission to view this page.</p>
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
              Batch Processing Management
            </h1>
            <p
              className="text-xs text-[#707070] mt-1"
              style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
            >
              Configure and monitor automated digitization pipelines for high-volume document buckets
            </p>
          </div>
          <Button
            variant="outline"
            onClick={handleRefreshAll}
            className="flex items-center gap-2 bg-white border-[#255c5d] text-[#255c5d] font-bold uppercase text-[10px] tracking-widest rounded-[4px] hover:bg-[#E8F3F3] h-10 px-6 transition-colors shadow-sm"
            style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
          >
            <RefreshCw className="h-3 w-3" />
            Synchronize Data
          </Button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Configuration Card */}
          <Card className="border border-[#E0E0E0] rounded-[8px] overflow-hidden shadow-sm flex flex-col">
            <div className="bg-[#F1F8F8] px-6 py-4 border-b border-[#E0E0E0]">
              <h2 className="text-[11px] font-bold text-[#255c5d] uppercase tracking-[0.2em]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>System Configuration</h2>
            </div>
            <CardContent className="p-6 space-y-6 flex-1">
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-[10px] font-bold text-[#707070] uppercase tracking-wider">Target GCS Bucket</Label>
                  <Select onValueChange={setSelectedBucket} value={selectedBucket}>
                    <SelectTrigger className="bg-white border-[#E0E0E0] h-12 rounded-[4px] text-sm text-[#000000] focus:ring-[#255c5d]" disabled={isLoadingBuckets || isLoadingConfig}>
                      <SelectValue placeholder={isLoadingBuckets || isLoadingConfig ? "Retrieving buckets..." : "Select a scanning bucket..."} />
                    </SelectTrigger>
                    <SelectContent>
                      {buckets?.map((bucketName) => (
                        <SelectItem key={bucketName} value={bucketName}>{bucketName}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex gap-3 pt-2">
                  <Button
                    onClick={handleSaveConfig}
                    disabled={saveConfigMutation.isPending || !selectedBucket}
                    className="flex-1 bg-[#255c5d] hover:bg-[#1E4E55] text-white font-bold uppercase text-[10px] tracking-widest rounded-[4px] h-11 transition-colors"
                  >
                    {saveConfigMutation.isPending ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                    Establish Default
                  </Button>

                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button
                        variant="outline"
                        disabled={!selectedBucket || startRunMutation.isPending}
                        className="flex-1 border-[#255c5d] text-[#255c5d] font-bold uppercase text-[10px] tracking-widest rounded-[4px] h-11 hover:bg-[#E8F3F3] transition-colors"
                      >
                        {startRunMutation.isPending ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                        Initiate Batch
                      </Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent className="rounded-[8px]">
                      <AlertDialogHeader>
                        <AlertDialogTitle className="text-[#255c5d]" style={{ fontFamily: 'var(--unnamed-font-family-georgia)' }}>Confirm Batch Execution</AlertDialogTitle>
                        <AlertDialogDescription className="text-sm text-[#707070]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>
                          Proceeding will initiate a full system scan of <strong>{selectedBucket}</strong>. New document assets will be queued for automated digitization.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel className="rounded-[4px] border-[#E0E0E0] text-[12px] uppercase font-bold tracking-widest">Cancel</AlertDialogCancel>
                        <AlertDialogAction onClick={handleStartRun} className="bg-[#255c5d] hover:bg-[#1E4E55] text-white rounded-[4px] text-[12px] uppercase font-bold tracking-widest transition-colors">Confirm Run</AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Ignored Folders Card */}
          <Card className="border border-[#E0E0E0] rounded-[8px] overflow-hidden shadow-sm flex flex-col">
            <div className="bg-[#F1F8F8] px-6 py-4 border-b border-[#E0E0E0]">
              <h2 className="text-[11px] font-bold text-[#255c5d] uppercase tracking-[0.2em]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Exclusion Management</h2>
            </div>
            <CardContent className="p-6 space-y-6 flex-1">
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-[10px] font-bold text-[#707070] uppercase tracking-wider">Exclude Folder Pattern</Label>
                  <div className="flex gap-2">
                    <Input
                      placeholder="e.g., archived_files/"
                      value={newIgnoredFolder}
                      onChange={(e) => setNewIgnoredFolder(e.target.value)}
                      className="bg-white border-[#E0E0E0] h-12 rounded-[4px] text-sm text-[#000000] focus:ring-[#255c5d] flex-1"
                    />
                    <Button
                      onClick={handleAddIgnoredFolder}
                      disabled={addIgnoredFolderMutation.isPending || !newIgnoredFolder.trim()}
                      className="bg-[#255c5d] hover:bg-[#1E4E55] text-white font-bold uppercase text-[10px] tracking-widest rounded-[4px] h-12 px-6 transition-colors"
                    >
                      {addIgnoredFolderMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : "Ignore"}
                    </Button>
                  </div>
                </div>

                <div className="mt-4 bg-[#F8F8F8] rounded-[6px] p-4 border border-[#E0E0E0]">
                  <h3 className="text-[10px] font-bold text-[#255c5d] uppercase tracking-wider mb-3">Excluded Directories</h3>
                  <div className="max-h-[120px] overflow-y-auto space-y-2">
                    {isLoadingIgnoredFolders ? (
                      <Loader2 className="h-4 w-4 animate-spin text-[#255c5d]/20 mx-auto" />
                    ) : ignoredFoldersData?.ignored_folders && ignoredFoldersData.ignored_folders.length > 0 ? (
                      ignoredFoldersData.ignored_folders.map((folder, index) => (
                        <div key={index} className="flex items-center gap-2 p-2 bg-white rounded border border-[#E0E0E0] group">
                          <FolderX className="h-3.5 w-3.5 text-red-400" />
                          <span className="text-xs font-mono text-[#707070] flex-1 truncate">{folder}</span>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-7 w-7 text-red-400 hover:text-red-500 hover:bg-red-50 opacity-0 group-hover:opacity-100 transition-opacity"
                            onClick={() => deleteIgnoredFolderMutation.mutate(folder)}
                            disabled={deleteIgnoredFolderMutation.isPending}
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      ))
                    ) : (
                      <p className="text-[10px] text-[#A0A0A0] text-center py-4 italic">No directories currently excluded from digitization runs</p>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Reports Card */}
        <Card className="border border-[#E0E0E0] rounded-[8px] overflow-hidden shadow-sm">
          <div className="bg-[#F1F8F8] px-6 py-4 border-b border-[#E0E0E0] flex justify-between items-center">
            <h2 className="text-[11px] font-bold text-[#255c5d] uppercase tracking-[0.2em]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Execution History & Analytics</h2>
            <div className="w-[300px]">
              <Select onValueChange={setSelectedRunId} value={selectedRunId}>
                <SelectTrigger className="bg-white border-[#E0E0E0] h-9 rounded-[4px] text-xs text-[#255c5d] focus:ring-[#255c5d] shadow-none" disabled={isLoadingReports}>
                  <SelectValue placeholder={isLoadingReports ? "Retrieving reports..." : "Select past execution..."} />
                </SelectTrigger>
                <SelectContent>
                  {reports?.map((report) => (
                    <SelectItem key={report.run_id} value={report.run_id}>
                      {new Date(report.start_timestamp._seconds * 1000).toLocaleString()} — {report.status}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <CardContent className="p-0">
            {selectedReport ? (
              <div className="p-8 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 bg-white border-b border-[#E0E0E0]">
                <div className="space-y-1 border-l-2 border-[#255c5d] pl-4">
                  <span className="text-[10px] font-bold text-[#707070] uppercase tracking-wider">Asset Status</span>
                  <div className="flex items-center gap-2">
                    <span className={cn(
                      "text-[10px] font-bold py-0.5 px-2 rounded-full border uppercase",
                      selectedReport.status === 'completed' ? "bg-green-50 text-green-600 border-green-200" : "bg-blue-50 text-blue-600 border-blue-200"
                    )}>
                      {selectedReport.status}
                    </span>
                  </div>
                </div>
                <div className="space-y-1 border-l-2 border-[#E0E0E0] pl-4">
                  <span className="text-[10px] font-bold text-[#707070] uppercase tracking-wider">Files Scanned</span>
                  <p className="text-xl font-bold text-[#255c5d]">{selectedReport.total_files_scanned.toLocaleString()}</p>
                </div>
                <div className="space-y-1 border-l-2 border-[#E0E0E0] pl-4">
                  <span className="text-[10px] font-bold text-[#707070] uppercase tracking-wider">New Assets Found</span>
                  <p className="text-xl font-bold text-[#255c5d]">{selectedReport.new_files_found.toLocaleString()}</p>
                </div>
                <div className="space-y-1 border-l-2 border-[#E0E0E0] pl-4">
                  <span className="text-[10px] font-bold text-[#707070] uppercase tracking-wider">Time Window</span>
                  <p className="text-[12px] font-medium text-[#707070]">
                    {new Date(selectedReport.start_timestamp._seconds * 1000).toLocaleTimeString()} —
                    {selectedReport.end_timestamp ? new Date(selectedReport.end_timestamp._seconds * 1000).toLocaleTimeString() : ' Active'}
                  </p>
                </div>
              </div>
            ) : null}

            {/* Processed Files Table */}
            {selectedRunId && (
              <div className="p-0">
                <Table>
                  <TableHeader className="bg-gray-50/50">
                    <TableRow className="border-b border-[#E0E0E0] hover:bg-transparent">
                      <TableHead className="py-4 pl-6 text-[11px] font-bold text-[#707070] uppercase tracking-wider">Digitized Asset Path</TableHead>
                      <TableHead className="py-4 text-[11px] font-bold text-[#707070] uppercase tracking-wider">Size (KB)</TableHead>
                      <TableHead className="py-4 text-[11px] font-bold text-[#707070] uppercase tracking-wider">Status</TableHead>
                      <TableHead className="py-4 text-[11px] font-bold text-[#707070] uppercase tracking-wider pr-6 text-right">Discovery Time</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {isLoadingFiles ? (
                      <TableRow>
                        <TableCell colSpan={4} className="py-20 text-center">
                          <Loader2 className="mx-auto h-8 w-8 animate-spin text-[#255c5d]/20" />
                        </TableCell>
                      </TableRow>
                    ) : processedFiles?.files && processedFiles.files.length > 0 ? (
                      processedFiles.files.map((file) => (
                        <TableRow key={file.file_gcs_path} className="border-b border-[#E0E0E0] hover:bg-gray-50 transition-colors">
                          <TableCell className="py-4 pl-6">
                            <div className="max-w-[500px] truncate font-mono text-[11px] text-[#707070]" title={file.file_gcs_path}>
                              {file.file_gcs_path}
                            </div>
                          </TableCell>
                          <TableCell className="py-4">
                            <span className="text-xs font-medium text-[#000000]">{(file.file_size / 1024).toFixed(2)}</span>
                          </TableCell>
                          <TableCell className="py-4">
                            <span className="text-[10px] font-bold uppercase py-0.5 px-2 bg-[#E8F3F3] text-[#255c5d] rounded-full">{file.status}</span>
                          </TableCell>
                          <TableCell className="py-4 pr-6 text-right">
                            <span className="text-[11px] font-medium text-[#707070]">
                              {new Date(file.first_seen_timestamp._seconds * 1000).toLocaleString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                            </span>
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={4} className="py-20 text-center italic text-[#707070] text-sm">No processed files listed for this execution run.</TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>

                {/* Pagination */}
                <div className="bg-gray-50/50 px-6 py-4 flex justify-between items-center border-t border-[#E0E0E0]">
                  <span className="text-[10px] font-bold text-[#707070] uppercase tracking-widest" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>
                    Asset Stream
                  </span>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setFilesCursor(null)}
                      disabled={!filesCursor}
                      className="h-8 border-[#E0E0E0] text-[10px] font-bold tracking-widest uppercase rounded-[4px]"
                    >
                      Reset
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setFilesCursor(processedFiles?.next_cursor || null)}
                      disabled={!processedFiles?.next_cursor}
                      className="h-8 border-[#E0E0E0] text-[10px] font-bold tracking-widest uppercase rounded-[4px] px-6"
                    >
                      Next Pipeline Segment
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
