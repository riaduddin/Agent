/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-unused-vars */
import React, { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Calendar } from "@/components/ui/calendar";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import {
  ArrowRightLeft,
  FileText,
  FolderOpen,
  Calendar as CalendarIcon,
  Filter,
  X,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  RefreshCw,
  Search,
  CheckCircle2,
  XCircle,
  AlertCircle,
  MoreVertical,
  Undo2,
  Loader2,
} from "lucide-react";
import { format } from "date-fns";
import axiosInstance from "@/lib/axiosInstance";
import { toast } from "sonner";
import { useAuth } from "@/context/AuthContext";
import { useFilePermissions } from "@/hooks/use-file-permissions";
import { cn } from "@/lib/utils";

// Types
interface Transfer {
  id: string;
  user_id: string;
  fileName: string;
  source: string;
  destination: string;
  operation: "move" | "undo" | "delete";
  status: "succeeded" | "failed" | "running" | "queued" | "partial";
  transfer_type: string | null;
  timestamp: string;
  can_undo: boolean;
}

interface PaginationInfo {
  current_page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
  total_items: number | null;
  total_pages: number | null;
}

interface TransferHistoryResponse {
  transfers: Transfer[];
  pagination: PaginationInfo;
  filters?: Record<string, any>;
}

// API Function
const fetchTransferLogs = async (params: {
  page: number;
  limit: number;
  status?: string;
  operation?: string;
  transferType?: string;
  startDate?: Date;
  endDate?: Date;
  date?: Date;
}): Promise<TransferHistoryResponse> => {
  const queryParams = new URLSearchParams();

  queryParams.append("page", params.page.toString());
  queryParams.append("limit", params.limit.toString());

  if (params.status && params.status !== "all") {
    queryParams.append("status", params.status);
  }

  if (params.operation && params.operation !== "all") {
    queryParams.append("operation", params.operation);
  }

  if (params.transferType && params.transferType !== "all") {
    queryParams.append("transfer_type", params.transferType);
  }

  // Handle date filtering - use either specific date OR date range
  if (params.date) {
    queryParams.append("date", format(params.date, "yyyy-MM-dd"));
  } else {
    if (params.startDate) {
      queryParams.append("start_date", format(params.startDate, "yyyy-MM-dd"));
    }
    if (params.endDate) {
      queryParams.append("end_date", format(params.endDate, "yyyy-MM-dd"));
    }
  }

  const response = await axiosInstance.get(
    `/gcs/user-transfers?${queryParams.toString()}`
  );

  return response.data;
};

const TransferHistory = () => {
  // Filter state
  const [status, setStatus] = useState<string>("all");
  const [operation, setOperation] = useState<string>("all");
  const [transferType, setTransferType] = useState<string>("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [startDate, setStartDate] = useState<Date | undefined>();
  const [endDate, setEndDate] = useState<Date | undefined>();

  // Track which transfer is currently being undone
  const [undoingTransferId, setUndoingTransferId] = useState<string | null>(null);

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);

  // Get user permissions
  const { canViewAllTransferHistory: canViewAll } = useFilePermissions();

  // Fetch data using TanStack Query
  const { data, isLoading, error, refetch, isFetching } = useQuery({
    queryKey: [
      "transferLogs",
      currentPage,
      itemsPerPage,
      status,
      operation,
      transferType,
      startDate,
      endDate,
    ],
    queryFn: () =>
      fetchTransferLogs({
        page: currentPage,
        limit: itemsPerPage,
        status,
        operation,
        transferType,
        startDate,
        endDate,
      }),
  });

  // Undo mutation
  const queryClient = useQueryClient();
  const undoMutation = useMutation({
    mutationFn: async (transferId: string) => {
      const response = await axiosInstance.post("/gcs/undo-transfer", {
        transfer_id: transferId,
      });
      return response.data;
    },
    onSuccess: () => {
      toast.success("Transfer undone successfully. File moved back to source.");
      setUndoingTransferId(null);
      // Invalidate and refetch transfer logs
      queryClient.invalidateQueries({ queryKey: ["transferLogs"] });
      // Also invalidate file tree data if available
      queryClient.invalidateQueries({ queryKey: ["gcsListPath"] });
      queryClient.invalidateQueries({ queryKey: ["gcsListDestinationPath"] });
    },
    onError: (error: any) => {
      setUndoingTransferId(null);
      const errorMessage = error?.response?.data?.error || error?.response?.data?.message || "Failed to undo transfer";
      toast.error(errorMessage);
    },
  });

  // Handle undo transfer
  const handleUndoTransfer = (transfer: Transfer) => {
    if (!transfer.can_undo) {
      toast.error("This transfer cannot be undone. The file may have already been processed.");
      return;
    }
    setUndoingTransferId(transfer.id);
    undoMutation.mutate(transfer.id);
  };

  // Apply search filter on frontend
  const filteredTransfers = useMemo(() => {
    if (!data?.transfers) return [];
    if (!searchTerm) return data.transfers;

    const term = searchTerm.toLowerCase();
    return data.transfers.filter(
      (transfer) =>
        transfer.fileName?.toLowerCase().includes(term) ||
        transfer.source?.toLowerCase().includes(term) ||
        transfer.destination?.toLowerCase().includes(term)
    );
  }, [data?.transfers, searchTerm]);

  // Handlers
  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  const handleClearFilters = () => {
    setStatus("all");
    setOperation("all");
    setTransferType("all");
    setSearchTerm("");
    setStartDate(undefined);
    setEndDate(undefined);
    setCurrentPage(1);
  };

  // Format date for display (e.g., 10/9/2025, 11:30 AM)
  const formatDateForTable = (dateString: string) => {
    try {
      return format(new Date(dateString), "M/d/yyyy, h:mm a");
    } catch {
      return dateString;
    }
  };

  // Get status display
  const getStatusDisplay = (status: string) => {
    switch (status?.toLowerCase()) {
      case "succeeded":
      case "success":
        return { text: "Success", dotColor: "bg-[#4CAF50]" };
      case "failed":
      case "error":
        return { text: "Failed", dotColor: "bg-[#F44336]" };
      case "running":
      case "queued":
      case "processing":
      case "partial":
        return { text: "Processing", dotColor: "bg-[#9C27B0]" };
      default:
        return { text: status || "Unknown", dotColor: "bg-gray-400" };
    }
  };

  const pagination = data?.pagination;

  return (
    <div className="space-y-8">
      {/* Page Header Area */}
      <div className="space-y-1">
        <h1
          className="text-[32px] font-bold text-[#202020]"
          style={{
            fontFamily: "var(--unnamed-font-family-georgia)",
            color: "var(--unnamed-color-1a1a1a)"
          }}
        >
          Transfer History
        </h1>
        <p className="text-[13px]" style={{
          fontFamily: "var(--unnamed-font-family-montserrat)",
          color: "var(--unnamed-color-707070)"
        }}>
          Track and monitor all file and folder transfer activities
        </p>
      </div>

      {/* Filters Area */}
      <div className="flex flex-col md:flex-row justify-between items-end gap-6">
        <div className="relative w-full md:w-[448px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-[16px] w-[16px] text-[#707070]" strokeWidth={2} style={{ color: "var(--unnamed-color-707070)" }} />
          <Input
            placeholder="Search"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 h-10 bg-white border-[#D0D4DC] rounded-[4px] text-[14px] text-[#000000] shadow-none font-medium focus-visible:ring-1 focus-visible:ring-[#165d5d]"
            style={{
              fontFamily: "var(--unnamed-font-family-montserrat)",
              borderColor: "var(--unnamed-color-707070)",
              color: "var(--unnamed-color-000000)"
            }}
          />
        </div>

        <div className="flex items-center gap-4">
          <div className="flex flex-col gap-1.5">
            <span className="text-[11px] font-bold uppercase tracking-wider pl-0.5" style={{
              fontFamily: "var(--unnamed-font-family-montserrat)",
              color: "var(--unnamed-color-202020)"
            }}>Start Date</span>
            <Popover>
              <PopoverTrigger asChild>
                <div className="relative cursor-pointer group">
                  <Input
                    readOnly
                    value={startDate ? format(startDate, "MM/dd/yyyy") : "MM/DD/YYYY"}
                    className={cn(
                      "w-[160px] h-10 pr-10 border-[#D0D4DC] rounded-[4px] text-[13px] text-[#000000] font-medium shadow-none focus-visible:ring-1 focus-visible:ring-[#165d5d] cursor-pointer",
                      !startDate && "text-[#707070]"
                    )}
                    style={{
                      fontFamily: "var(--unnamed-font-family-montserrat)",
                      borderColor: "var(--unnamed-color-707070)"
                    }}
                  />
                  <CalendarIcon className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4" style={{ color: "var(--unnamed-color-707070)" }} />
                </div>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0" align="end">
                <Calendar
                  mode="single"
                  selected={startDate}
                  onSelect={setStartDate}
                  initialFocus
                />
              </PopoverContent>
            </Popover>
          </div>

          <div className="flex flex-col gap-1.5">
            <span className="text-[11px] font-bold uppercase tracking-wider pl-0.5" style={{
              fontFamily: "var(--unnamed-font-family-montserrat)",
              color: "var(--unnamed-color-202020)"
            }}>End Date</span>
            <Popover>
              <PopoverTrigger asChild>
                <div className="relative cursor-pointer group">
                  <Input
                    readOnly
                    value={endDate ? format(endDate, "MM/dd/yyyy") : "MM/DD/YYYY"}
                    className={cn(
                      "w-[160px] h-10 pr-10 border-[#D0D4DC] rounded-[4px] text-[13px] text-[#000000] font-medium shadow-none focus-visible:ring-1 focus-visible:ring-[#165d5d] cursor-pointer",
                      !endDate && "text-[#707070]"
                    )}
                    style={{
                      fontFamily: "var(--unnamed-font-family-montserrat)",
                      borderColor: "var(--unnamed-color-707070)"
                    }}
                  />
                  <CalendarIcon className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4" style={{ color: "var(--unnamed-color-707070)" }} />
                </div>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0" align="end">
                <Calendar
                  mode="single"
                  selected={endDate}
                  onSelect={setEndDate}
                  initialFocus
                />
              </PopoverContent>
            </Popover>
          </div>
        </div>
      </div>

      {/* Table Section */}
      <div className="bg-white border rounded-[8px] overflow-hidden shadow-sm" style={{ borderColor: "var(--unnamed-color-e2eae4)" }}>
        <Table>
          <TableHeader style={{ backgroundColor: "var(--unnamed-color-eef4f4)" }}>
            <TableRow className="border-b hover:bg-transparent" style={{ borderColor: "var(--unnamed-color-e2eae4)" }}>
              <TableHead className="text-[12px] font-medium py-3 pl-6 w-[56px] uppercase tracking-tight" style={{ color: "var(--unnamed-color-1e4e55)" }}>Type</TableHead>
              <TableHead className="text-[12px] font-medium py-3 uppercase tracking-tight" style={{ color: "var(--unnamed-color-1e4e55)" }}>
                <div className="flex items-center gap-2">
                  Item <Filter className="h-[14px] w-[14px]" />
                </div>
              </TableHead>
              <TableHead className="text-[12px] font-medium py-3 uppercase tracking-tight" style={{ color: "var(--unnamed-color-1e4e55)" }}>
                <div className="flex items-center gap-2">
                  Source Path <Filter className="h-[14px] w-[14px]" />
                </div>
              </TableHead>
              <TableHead className="text-[12px] font-medium py-3 uppercase tracking-tight" style={{ color: "var(--unnamed-color-1e4e55)" }}>
                <div className="flex items-center gap-2">
                  Destination Path <Filter className="h-[14px] w-[14px]" />
                </div>
              </TableHead>
              <TableHead className="text-[12px] font-medium py-3 uppercase tracking-tight" style={{ color: "var(--unnamed-color-1e4e55)" }}>
                <div className="flex items-center gap-2">
                  Status <Filter className="h-[14px] w-[14px]" />
                </div>
              </TableHead>
              <TableHead className="text-[12px] font-medium py-3 uppercase tracking-tight" style={{ color: "var(--unnamed-color-1e4e55)" }}>
                <div className="flex items-center gap-2">
                  Timestamp <Filter className="h-[14px] w-[14px]" />
                </div>
              </TableHead>
              <TableHead className="text-[12px] font-medium py-3 text-center uppercase tracking-tight" style={{ color: "var(--unnamed-color-1e4e55)" }}>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              Array.from({ length: 5 }).map((_, i) => (
                <TableRow key={i} className="h-[64px]">
                  <TableCell className="pl-6"><Skeleton className="h-[18px] w-[18px]" /></TableCell>
                  <TableCell><Skeleton className="h-4 w-32" /></TableCell>
                  <TableCell><Skeleton className="h-4 w-48" /></TableCell>
                  <TableCell><Skeleton className="h-4 w-48" /></TableCell>
                  <TableCell><Skeleton className="h-4 w-20" /></TableCell>
                  <TableCell><Skeleton className="h-4 w-32" /></TableCell>
                  <TableCell><Skeleton className="h-5 w-5 mx-auto" /></TableCell>
                </TableRow>
              ))
            ) : filteredTransfers.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} className="h-32 text-center" style={{ color: "var(--unnamed-color-707070)" }}>
                  No transfer history found matching your criteria.
                </TableCell>
              </TableRow>
            ) : (
              filteredTransfers.map((transfer) => {
                const statusInfo = getStatusDisplay(transfer.status);
                return (
                  <TableRow key={transfer.id} className="border-b transition-colors h-[64px]" style={{ borderColor: "var(--unnamed-color-e2eae4)" }}>
                    <TableCell className="pl-6">
                      {transfer.transfer_type?.toLowerCase() === "folder" ? (
                        <FolderOpen className="h-[18px] w-[18px]" style={{ color: "var(--unnamed-color-707070)" }} />
                      ) : (
                        <FileText className="h-[18px] w-[18px]" style={{ color: "var(--unnamed-color-707070)" }} />
                      )}
                    </TableCell>
                    <TableCell>
                      <div className="flex flex-col gap-1 py-1">
                        <span className="text-[13px] font-medium" style={{
                          fontFamily: "var(--unnamed-font-family-montserrat)",
                          color: "var(--unnamed-color-1a1a1a)"
                        }}>
                          {transfer.fileName || "Unknown"}
                        </span>
                        <div className="flex">
                          {transfer.operation === "move" ? (
                            <span className="text-[10px] font-bold px-[6px] py-[1px] rounded-[2px] uppercase tracking-[0.05em]" style={{
                              backgroundColor: "var(--color-move-bg)",
                              color: "var(--color-move-text)"
                            }}>MOVE</span>
                          ) : transfer.operation === "undo" ? (
                            <span className="text-[10px] font-bold px-[6px] py-[1px] rounded-[2px] uppercase tracking-[0.05em]" style={{
                              backgroundColor: "var(--color-undo-tag-bg)",
                              color: "var(--color-undo-tag-text)"
                            }}>UNDO</span>
                          ) : (
                            <span className="text-[10px] font-bold px-[6px] py-[1px] rounded-[2px] uppercase tracking-[0.05em]" style={{
                              backgroundColor: "var(--unnamed-color-f8f8f8)",
                              color: "var(--unnamed-color-707070)"
                            }}>{transfer.operation}</span>
                          )}
                        </div>
                      </div>
                    </TableCell>
                    <TableCell>
                      <span className="text-[13px] truncate max-w-[220px] block" title={transfer.source} style={{
                        fontFamily: "var(--unnamed-font-family-montserrat)",
                        color: "var(--unnamed-color-1a1a1a)"
                      }}>
                        {transfer.source || "-"}
                      </span>
                    </TableCell>
                    <TableCell>
                      <span className="text-[13px] truncate max-w-[220px] block" title={transfer.destination} style={{
                        fontFamily: "var(--unnamed-font-family-montserrat)",
                        color: "var(--unnamed-color-1a1a1a)"
                      }}>
                        {transfer.destination || "-"}
                      </span>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <div className={cn("w-[8px] h-[8px] rounded-full", statusInfo.dotColor)} />
                        <span className="text-[13px] font-medium" style={{
                          fontFamily: "var(--unnamed-font-family-montserrat)",
                          color: "var(--unnamed-color-1a1a1a)"
                        }}>
                          {statusInfo.text}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <span className="text-[13px]" style={{
                        fontFamily: "var(--unnamed-font-family-montserrat)",
                        color: "var(--unnamed-color-1a1a1a)"
                      }}>
                        {formatDateForTable(transfer.timestamp)}
                      </span>
                    </TableCell>
                    <TableCell className="text-center">
                      <TooltipProvider delayDuration={100}>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-8 w-8 p-0 disabled:opacity-30"
                              onClick={() => handleUndoTransfer(transfer)}
                              disabled={!transfer.can_undo || undoingTransferId === transfer.id}
                              style={{ color: "var(--unnamed-color-1e4e55)" }}
                            >
                              {undoingTransferId === transfer.id ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : (
                                <Undo2 className="h-5 w-5" strokeWidth={2.5} />
                              )}
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent side="left" className="border-none text-[11px] font-bold px-2 py-1" style={{
                            backgroundColor: "var(--unnamed-color-1e4e55)",
                            color: "var(--unnamed-color-ffffff)"
                          }}>
                            {transfer.can_undo ? "Undo" : "Cannot be undone"}
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </TableCell>
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>

        {/* Footer / Pagination Section */}
        <div className="px-6 py-4 bg-white border-t flex flex-col md:flex-row justify-end items-center gap-[40px]" style={{ borderColor: "var(--unnamed-color-e2eae4)" }}>
          <div className="flex items-center gap-3">
            <span className="text-[13px]" style={{
              fontFamily: "var(--unnamed-font-family-montserrat)",
              color: "var(--unnamed-color-000000)"
            }}>Items per page:</span>
            <Select value={String(itemsPerPage)} onValueChange={(value) => setItemsPerPage(Number(value))}>
              <SelectTrigger className="w-[72px] h-[32px] rounded-[4px] text-[13px] shadow-none font-medium" style={{ borderColor: "var(--unnamed-color-707070)" }}>
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
            <span className="text-[13px]" style={{
              fontFamily: "var(--unnamed-font-family-montserrat)",
              color: "var(--unnamed-color-000000)"
            }}>
              {pagination?.total_items ? (
                `${(currentPage - 1) * itemsPerPage + 1} - ${Math.min(currentPage * itemsPerPage, pagination.total_items)} of ${pagination.total_items}`
              ) : "1 - 0 of 0"}
            </span>

            <div className="flex items-center gap-2">
              <button
                onClick={() => handlePageChange(1)}
                disabled={currentPage <= 1 || isLoading}
                className="h-8 w-8 flex items-center justify-center disabled:opacity-30 hover:bg-gray-100 rounded transition-colors"
                style={{ color: "var(--unnamed-color-707070)" }}
              >
                <ChevronsLeft className="h-[20px] w-[20px]" strokeWidth={1.5} />
              </button>
              <button
                onClick={() => handlePageChange(currentPage - 1)}
                disabled={!pagination?.has_prev || isLoading}
                className="h-8 w-8 flex items-center justify-center disabled:opacity-30 hover:bg-gray-100 rounded transition-colors"
                style={{ color: "var(--unnamed-color-707070)" }}
              >
                <ChevronLeft className="h-[20px] w-[20px]" strokeWidth={1.5} />
              </button>
              <button
                onClick={() => handlePageChange(currentPage + 1)}
                disabled={!pagination?.has_next || isLoading}
                className="h-8 w-8 flex items-center justify-center disabled:opacity-30 hover:bg-gray-100 rounded transition-colors"
                style={{ color: "var(--unnamed-color-707070)" }}
              >
                <ChevronRight className="h-[20px] w-[20px]" strokeWidth={1.5} />
              </button>
              <button
                onClick={() => handlePageChange(pagination?.total_pages || 1)}
                disabled={currentPage >= (pagination?.total_pages || 1) || isLoading}
                className="h-8 w-8 flex items-center justify-center disabled:opacity-30 hover:bg-gray-100 rounded transition-colors"
                style={{ color: "var(--unnamed-color-707070)" }}
              >
                <ChevronsRight className="h-[20px] w-[20px]" strokeWidth={1.5} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TransferHistory;
