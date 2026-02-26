"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import axiosInstance from "@/lib/axiosInstance";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
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
import { Play, Loader2, CheckCircle2, RotateCcw, FileCheck, Clock, ArrowLeft } from "lucide-react";
import { toast } from "react-hot-toast";

const fetchStatus = async () => {
  const response = await axiosInstance.get("/batch/system/categorization-batch-status");
  return response.data;
};

const startBackfill = async () => {
  const response = await axiosInstance.post("/batch/system/start-categorization-batch");
  return response.data;
};

const resetStatus = async () => {
  const response = await axiosInstance.post("/batch/system/reset-categorization-batch-status");
  return response.data;
};

const fetchCategoryDistribution = async () => {
  const response = await axiosInstance.get("/batch/system/category-distribution");
  return response.data;
};

export default function CategoryBackfillPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [isStarting, setIsStarting] = useState(false);
  const [isResetting, setIsResetting] = useState(false);

  const { data: statusData, isLoading, isError } = useQuery({
    queryKey: ["categorizationStatus"],
    queryFn: fetchStatus,
    refetchInterval: 5000, // Poll every 5 seconds
  });

  const { data: distributionData, isLoading: isDistributionLoading } = useQuery({
    queryKey: ["categoryDistribution"],
    queryFn: fetchCategoryDistribution,
    refetchInterval: 5000,
  });

  const startMutation = useMutation({
    mutationFn: startBackfill,
    onMutate: () => {
      setIsStarting(true);
    },
    onSuccess: () => {
      toast.success("Categorization backfill started successfully!");
      queryClient.invalidateQueries({ queryKey: ["categorizationStatus"] });
      queryClient.invalidateQueries({ queryKey: ["categoryDistribution"] });
    },
    onError: (error: any) => {
      const msg = error.response?.data?.msg || "Failed to start backfill";
      toast.error(msg);
    },
    onSettled: () => {
      setIsStarting(false);
    },
  });

  const resetMutation = useMutation({
    mutationFn: resetStatus,
    onMutate: () => {
      setIsResetting(true);
    },
    onSuccess: () => {
      toast.success("Status reset to idle.");
      queryClient.invalidateQueries({ queryKey: ["categorizationStatus"] });
      queryClient.invalidateQueries({ queryKey: ["categoryDistribution"] });
    },
    onError: (error: any) => {
      toast.error("Failed to reset status");
    },
    onSettled: () => {
      setIsResetting(false);
    },
  });

  const isRunning = statusData?.status === "running";

  return (
    <div className="container mx-auto py-8 space-y-8">
      <div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => router.back()}
          className="mb-2 pl-0 hover:bg-transparent hover:text-primary"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back
        </Button>
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Category Backfill Tool</h1>
            <p className="text-muted-foreground mt-2">
              Trigger and monitor the background categorization process for all documents.
            </p>
          </div>
          <div className="flex gap-2">
            <Button
              onClick={() => resetMutation.mutate()}
              disabled={isResetting || isLoading}
              variant="outline"
              size="lg"
            >
              {isResetting ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <RotateCcw className="mr-2 h-4 w-4" />
              )}
              Reset Status
            </Button>
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button
                  disabled={isRunning || isStarting || isLoading}
                  size="lg"
                  className={isRunning ? "bg-green-600 hover:bg-green-700" : ""}
                >
                  {isStarting ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : isRunning ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <Play className="mr-2 h-4 w-4" />
                  )}
                  {isRunning ? "Running..." : "Start Backfill"}
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Are you sure?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This will trigger the category backfill process for ALL documents in the system.
                    <br /><br />
                    - It scans every document in the database.
                    - Documents without categories will be processed by AI.
                    - This runs in the background but may create a high load.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={() => startMutation.mutate()}>
                    Confirm Start
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Status</CardTitle>
            {isRunning ? (
              <Badge variant="default" className="bg-green-500">Running</Badge>
            ) : (
              <Badge variant="secondary">Idle</Badge>
            )}
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="animate-pulse h-8 w-24 bg-muted rounded" />
            ) : isError ? (
              <div className="text-2xl font-bold text-red-500">Offline</div>
            ) : (
              <div className="text-2xl font-bold capitalize">{statusData?.status || "Unknown"}</div>
            )}
            <p className="text-xs text-muted-foreground">
              Current system status
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Progress</CardTitle>
            <FileCheck className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="animate-pulse h-8 w-16 bg-muted rounded" />
            ) : isError ? (
              <div className="text-2xl font-bold text-red-500">-</div>
            ) : (
              <div className="flex items-baseline gap-2">
                <div className="text-2xl font-bold">{statusData?.processed_count?.toLocaleString() || 0}</div>
                <span className="text-sm text-muted-foreground">/ {statusData?.total_documents?.toLocaleString() || 0}</span>
              </div>
            )}
            <p className="text-xs text-muted-foreground">
              Scanned / Total Documents
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Pending</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="animate-pulse h-8 w-16 bg-muted rounded" />
            ) : isError ? (
              <div className="text-2xl font-bold text-red-500">-</div>
            ) : (
              <div className="text-2xl font-bold">{statusData?.pending_count?.toLocaleString() || 0}</div>
            )}
            <p className="text-xs text-muted-foreground">
              Remaining documents
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Categorized</CardTitle>
            <CheckCircle2 className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="animate-pulse h-8 w-16 bg-muted rounded" />
            ) : isError ? (
              <div className="text-2xl font-bold text-red-500">-</div>
            ) : (
              <div className="text-2xl font-bold text-green-600">{statusData?.actioned_count?.toLocaleString() || 0}</div>
            )}
            <p className="text-xs text-muted-foreground">
              Successfully updated in this run
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Category Distribution</CardTitle>
          <CardDescription>Breakdown of documents by category</CardDescription>
        </CardHeader>
        <CardContent className="max-h-[400px] overflow-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[80px]">Serial</TableHead>
                <TableHead>Category</TableHead>
                <TableHead className="text-right">File Count</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isDistributionLoading ? (
                <TableRow>
                  <TableCell colSpan={3} className="text-center py-8">
                    <Loader2 className="h-6 w-6 animate-spin mx-auto text-muted-foreground" />
                  </TableCell>
                </TableRow>
              ) : distributionData?.length > 0 ? (
                distributionData.map((item: any, index: number) => (
                  <TableRow key={index}>
                    <TableCell className="font-medium text-muted-foreground">{index + 1}</TableCell>
                    <TableCell>
                      <Badge variant="outline" className="font-mono">{item.category}</Badge>
                    </TableCell>
                    <TableCell className="text-right font-bold">{item.count}</TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={3} className="text-center py-8 text-muted-foreground">
                    No data available
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Recent Activity</CardTitle>
          <CardDescription>
            Live log of categorization events from the background worker.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Time</TableHead>
                <TableHead>Document ID</TableHead>
                <TableHead>Action</TableHead>
                <TableHead>Details</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading ? (
                <TableRow>
                  <TableCell colSpan={4} className="text-center py-8 text-muted-foreground">
                    <div className="flex justify-center items-center space-x-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span>Loading activity log...</span>
                    </div>
                  </TableCell>
                </TableRow>
              ) : isError ? (
                <TableRow>
                  <TableCell colSpan={4} className="text-center py-4 text-red-500">
                    Failed to load recent activity.
                  </TableCell>
                </TableRow>
              ) : statusData?.recent_activity?.length > 0 ? (
                statusData.recent_activity.map((log: any, i: number) => (
                  <TableRow key={i}>
                    <TableCell className="font-mono text-xs">
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </TableCell>
                    <TableCell className="font-mono text-xs truncate max-w-[150px]" title={log.doc_id}>
                      {log.doc_id}
                    </TableCell>
                    <TableCell>
                      <Badge variant={log.action === "categorization_finished" ? "outline" : "secondary"}>
                        {log.action?.replace("categorization_", "")}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground truncate max-w-[300px]">
                      {JSON.stringify(log.details)}
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={4} className="text-center py-4 text-muted-foreground">
                    No recent activity found.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
