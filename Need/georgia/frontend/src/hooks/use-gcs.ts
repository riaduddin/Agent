/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/no-explicit-any */
import api from "@/lib/axiosInstance";
import {
  useQuery,
  useMutation,
  useQueryClient,
  useInfiniteQuery,
  keepPreviousData,
} from "@tanstack/react-query";
import { toast } from "sonner";

// Types
export interface GCSFolder {
  name: string;
  updatedAt: null;
}

export interface GCSFile {
  name: string;
  size: number;
  updatedAt: string;
  contentType: string;
}

export interface GCSPagination {
  current_page: number;
  total_pages: number;
  page_size: number;
  total_items: number;
  total_folders: number;
  total_files: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface GCSListResponse {
  path: string;
  folders: GCSFolder[];
  files: GCSFile[];
  pagination: GCSPagination;
}

export interface TransferProgress {
  transfer_id: string;
  type: "folder" | "file";
  operation: "move" | "copy";
  status: "pending" | "in_progress" | "completed" | "failed";
  total_items: number;
  completed_items: number;
  current_item: string;
  start_time: string;
  end_time: string | null;
  error: string | null;
}

export interface FileEligibility {
  can_delete: boolean;
  can_undo: boolean;
  destination_path: string;
  is_batch_processed: boolean;
}

export interface UndoTransferResponse {
  destination_path: string;
  message: string;
  source_path: string;
}

export interface DeleteDestinationFileResponse {
  file_path: string;
  message: string;
}

export interface TransferConflictResponse {
  conflicts: string[];
}

// Query Hooks
export function useGCSListRoot(page: number = 1, pageSize: number = 50) {
  return useQuery({
    queryKey: ["gcs", "list-root", page, pageSize],
    queryFn: async () => {
      const params = new URLSearchParams({
        page: page.toString(),
        page_size: pageSize.toString(),
      });

      const { data } = await api.get<GCSListResponse>(
        `/gcs/list-root?${params}`
      );
      return data;
    },
  });
}

export function useGCSListWithSubfolders(
  page: number = 1,
  pageSize: number = 50,
  searchQuery?: string
) {
  return useQuery({
    queryKey: ["gcs", "list-with-subfolders", page, pageSize, searchQuery],
    queryFn: async () => {
      const params = new URLSearchParams({
        page: page.toString(),
        page_size: pageSize.toString(),
        search: searchQuery?.trim() || "",
        path: "Georgia 14/Pending Files/",
      });

      const { data } = await api.get<any>(
        `/gcs/list-with-subfolders?${params}`
      );
      return data;
    },
  });
}

export function useGCSDestinationsWithSubfolders(
  page: number = 1,
  pageSize: number = 50,
  searchQuery?: string
) {
  return useQuery({
    queryKey: [
      "gcs",
      "destinations-with-subfolders",
      page,
      pageSize,
      searchQuery,
    ],
    queryFn: async () => {
      const params = new URLSearchParams({
        page: page.toString(),
        page_size: pageSize.toString(),
        search: searchQuery?.trim() || "",
      });

      const { data } = await api.get<any>(
        `/gcs/destinations-with-subfolders?${params}`
      );
      return data;
    },
  });
}

export function useGCSListPath(
  path: string,
  page: number = 1,
  pageSize: number = 50
) {
  return useQuery({
    queryKey: ["gcs", "list-path", path, page, pageSize],
    queryFn: async () => {
      const params = new URLSearchParams({
        path: path,
        page: page.toString(),
        page_size: pageSize.toString(),
      });

      const { data } = await api.get<GCSListResponse>(
        `/gcs/list-path?${params}`
      );
      return data;
    },
    enabled: !!path, // Only run if path is provided
  });
}

export function useTransferProgress(transferId: string | null) {
  return useQuery({
    queryKey: ["gcs", "transfer-progress", transferId],
    queryFn: async () => {
      if (!transferId) return null;
      const { data } = await api.get<TransferProgress>(
        `/gcs/progress/${transferId}`
      );
      return data;
    },
    enabled: !!transferId,
    refetchInterval: (result) => {
      // Poll every 1 second if transfer is in progress
      if (
        result.state.data?.status === "in_progress" ||
        result.state.data?.status === "pending"
      ) {
        return 1000;
      }
      return false;
    },
  });
}

// Mutation Hooks
export function useCreateFolder() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (input: { folderName: string; currentPath: string }) => {
      const { data } = await api.post("/gcs/create-folder", input);
      return data;
    },
    onSuccess: (data, variables) => {
      // Invalidate list queries for the parent path
      queryClient.invalidateQueries({ queryKey: ["gcs", "list-root"] });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-with-subfolders"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "destinations-with-subfolders"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-path", variables.currentPath],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-path"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-destination-path"],
      });
      queryClient.invalidateQueries({
        queryKey: ["transferLogs"],
      });
      toast.success(data.message || "Folder created successfully");
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.message || "Failed to create folder");
    },
  });
}

export function useUploadFiles() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (input: { path?: string; files: File[] }) => {
      const formData = new FormData();

      if (input.path) {
        formData.append("path", input.path);
      }

      input.files.forEach((file) => {
        formData.append("files", file);
      });

      const { data } = await api.post("/gcs/upload-files-to-path", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      return data;
    },
    onSuccess: (data, variables) => {
      // Invalidate list queries
      queryClient.invalidateQueries({ queryKey: ["gcs"] });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-with-subfolders"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "destinations-with-subfolders"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-path"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-destination-path"],
      });
      queryClient.invalidateQueries({
        queryKey: ["transferLogs"],
      });
      toast.success(data.message || "Files uploaded successfully");
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.message || "Failed to upload files");
    },
  });
}

export function useUploadFilesWithConflict() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (input: { path?: string; files: File[] }) => {
      const formData = new FormData();

      if (input.path) {
        formData.append("path", input.path);
      }

      input.files.forEach((file) => {
        formData.append("files", file);
      });

      const { data } = await api.post("/gcs/upload-files-to-path", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      return data;
    },
    onSuccess: (data, variables) => {
      // Invalidate list queries
      queryClient.invalidateQueries({ queryKey: ["gcs"] });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-with-subfolders"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "destinations-with-subfolders"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-path"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-destination-path"],
      });
      queryClient.invalidateQueries({
        queryKey: ["transferLogs"],
      });

      // If there are errors (multi-status 207), show warning
      if (data.errors && data.errors.length > 0) {
        toast.warning(`Uploaded ${data.total_uploaded} files. ${data.total_errors} failed.`);
      } else {
        toast.success(data.message || "Files uploaded successfully");
      }
    },
    onError: (error: any) => {
      // 409 Conflict is handled by the component
      if (error.response?.status !== 409) {
        if (error.response?.data?.errors && Array.isArray(error.response.data.errors)) {
          // Display the first error or join them if multiple
          toast.error(error.response.data.errors.join("\n"));
        } else {
          toast.error(error.response?.data?.message || "Failed to upload files");
        }
      }
    },
  });
}

export function useResolveUploadConflict() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (input: {
      filePath: string;
      action: "keep" | "replace";
      targetPath?: string;
      file?: File
    }) => {
      if (input.action === 'replace' && input.file) {
        // Use multipart/form-data for replace
        const formData = new FormData();
        formData.append("filePath", input.filePath);
        formData.append("action", input.action);
        if (input.targetPath) {
          formData.append("targetPath", input.targetPath);
        }
        formData.append("file", input.file);

        const { data } = await api.post("/gcs/resolve-upload-conflict", formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });
        return data;
      } else {
        // Use JSON for keep
        const { data } = await api.post("/gcs/resolve-upload-conflict", {
          filePath: input.filePath,
          action: input.action,
          targetPath: input.targetPath
        });
        return data;
      }
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["gcs"] });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-with-subfolders"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "destinations-with-subfolders"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-path"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-destination-path"],
      });

      if (data.action === 'replace') {
        toast.success("File replaced successfully");
      } else {
        toast.info("Existing file kept");
      }
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.message || "Failed to resolve conflict");
    },
  });
}

export function useCheckTransferConflicts() {
  return useMutation({
    mutationFn: async (input: {
      source_paths: string[];
      destination: { root: string; path?: string };
    }) => {
      const { data } = await api.post<TransferConflictResponse>("/gcs/check-transfer-conflicts", input);
      return data;
    },
  });
}

export function useTransferFile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (input: {
      source: { path: string };
      destination: { root: string; path?: string };
      operation: "move" | "copy" | null;
    }) => {
      const { data } = await api.post("/gcs/transfer", input);
      return data;
    },
    onSuccess: (data) => {
      // Immediate invalidation
      queryClient.invalidateQueries({ queryKey: ["gcs"] });

      // Delayed invalidation to handle GCS eventual consistency
      setTimeout(() => {
        queryClient.invalidateQueries({ queryKey: ["gcs"] });
      }, 2000);

      toast.success(
        data.status === "succeeded"
          ? "File transferred successfully"
          : "File transfer initiated"
      );
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.message || "Failed to transfer file");
    },
  });
}

export function useTransferFolder() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (input: {
      sourceFolderPath: string;
      destination: { root: string; path?: string };
      operation?: "move" | "copy" | null;
    }) => {
      const { data } = await api.post("/gcs/transfer-folder", input);
      return data;
    },
    onSuccess: (data) => {
      // Immediate invalidation
      queryClient.invalidateQueries({ queryKey: ["gcs"] });

      // Delayed invalidation to handle GCS eventual consistency
      setTimeout(() => {
        queryClient.invalidateQueries({ queryKey: ["gcs"] });
      }, 2000);

      toast.success(
        `Folder transfer started: ${data.totalFiles} files to transfer`
      );
      return data;
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.message || "Failed to transfer folder");
    },
  });
}

interface DeleteResult {
  successful: string[];
  failed: { path: string; error: string }[];
}

export function useDeleteGCS() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (paths: string[]): Promise<DeleteResult> => {
      // Separate files and folders for batch processing
      const folders = paths.filter((p) => p.endsWith("/"));
      const files = paths.filter((p) => !p.endsWith("/"));

      const results: DeleteResult = {
        successful: [],
        failed: [],
      };

      // Process deletions in parallel using Promise.allSettled
      const deletePromises = [
        ...files.map((path) =>
          api
            .delete("/gcs/delete-file", { data: { filePath: path } })
            .then(() => ({ success: true, path }))
            .catch((error: any) => ({
              success: false,
              path,
              error:
                error.response?.data?.message ||
                error.message ||
                "Unknown error",
            }))
        ),
        ...folders.map((path) =>
          api
            .delete("/gcs/delete-folder", { data: { folderPath: path } })
            .then(() => ({ success: true, path }))
            .catch((error: any) => ({
              success: false,
              path,
              error:
                error.response?.data?.message ||
                error.message ||
                "Unknown error",
            }))
        ),
      ];

      const settled = await Promise.allSettled(deletePromises);

      // Process results
      settled.forEach((result) => {
        if (result.status === "fulfilled") {
          const { success, path } = result.value;
          if (success) {
            results.successful.push(path);
          } else {
            const error =
              "error" in result.value ? result.value.error : "Unknown error";
            results.failed.push({ path, error: error });
          }
        } else {
          results.failed.push({
            path: "unknown",
            error: result.reason?.message || "Request failed",
          });
        }
      });

      return results;
    },
    onSuccess: (results) => {
      // Invalidate queries only once after all deletions
      queryClient.invalidateQueries({ queryKey: ["gcs"] });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-with-subfolders"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "destinations-with-subfolders"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-path"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-destination-path"],
      });
      queryClient.invalidateQueries({
        queryKey: ["transferLogs"],
      });

      // Show consolidated toast message
      const total = results.successful.length + results.failed.length;

      if (results.failed.length === 0) {
        toast.success(
          `Successfully deleted ${results.successful.length} item(s)`
        );
      } else if (results.successful.length === 0) {
        toast.error(`Failed to delete all ${total} item(s)`);
      } else {
        toast.warning(
          `Deleted ${results.successful.length} of ${total} item(s). ${results.failed.length} failed.`
        );
      }
    },
    onError: (error: any) => {
      toast.error(
        error.response?.data?.message ||
        error.message ||
        "Failed to delete items"
      );
    },
  });
}

export function useFilePreview() {
  return useMutation({
    mutationFn: async (filePath: string) => {
      const encodedPath = encodeURIComponent(filePath);
      const { data } = await api.get(`/gcs/preview/${encodedPath}`);
      return data;
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.message || "Failed to load preview");
    },
  });
}

export const useRenameFolder = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      folderPath,
      newFolderName,
    }: {
      folderPath: string;
      newFolderName: string;
    }) => {
      const { data } = await api.patch("/gcs/rename-folder", {
        folderPath,
        newFolderName,
      });
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["gcs"] });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-with-subfolders"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "destinations-with-subfolders"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-path"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-destination-path"],
      });
      queryClient.invalidateQueries({
        queryKey: ["transferLogs"],
      });
      toast.success("Folder renamed successfully");
    },
    onError: (error: any) => {
      const message =
        error?.response?.data?.message ||
        error?.message ||
        "Failed to rename folder";
      toast.error(`Failed to rename folder: ${message}`);
    },
  });
};

export const useRenameFile = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      filePath,
      newFileName,
    }: {
      filePath: string;
      newFileName: string;
    }) => {
      const { data } = await api.patch("/gcs/rename-file", {
        filePath,
        newFileName,
      });
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["gcs"] });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-with-subfolders"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "destinations-with-subfolders"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-path"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-destination-path"],
      });
      queryClient.invalidateQueries({
        queryKey: ["transferLogs"],
      });
      toast.success("File renamed successfully");
    },
    onError: (error: any) => {
      const message =
        error?.response?.data?.message ||
        error?.message ||
        "Failed to rename file";
      toast.error(`Failed to rename file: ${message}`);
    },
  });
};

export function useGCSListPathInfinite(
  path: string,
  pageSize: number = 50,
  searchQuery?: string,
  fetchParents: boolean = true
) {
  return useInfiniteQuery({
    queryKey: ["gcs", "list-path", path, pageSize, searchQuery, fetchParents],
    queryFn: async ({ pageParam = 1 }) => {
      const params = new URLSearchParams({
        page: pageParam.toString(),
        page_size: pageSize.toString(),
        fetch_parents: fetchParents.toString(),
      });

      if (searchQuery?.trim()) {
        params.append("search", searchQuery.trim());
      }

      const { data } = await api.get<any>(
        `/gcs/dynamic-source-path?path=${encodeURIComponent(
          path
        )}&${params.toString()}`
      );
      return data;
    },
    getNextPageParam: (lastPage) => {
      return lastPage.pagination?.has_next
        ? lastPage.pagination.current_page + 1
        : undefined;
    },
    initialPageParam: 1,
    enabled: !!path,
    placeholderData: keepPreviousData,
  });
}

export function useGCSListDestinationPathInfinite(
  path: string,
  pageSize: number = 50,
  searchQuery?: string,
  fetchParents: boolean = true
) {
  return useInfiniteQuery({
    queryKey: ["gcs", "list-destination-path", path, pageSize, searchQuery, fetchParents],
    queryFn: async ({ pageParam = 1 }) => {
      const params = new URLSearchParams({
        page: pageParam.toString(),
        page_size: pageSize.toString(),
        fetch_parents: fetchParents.toString(),
      });

      if (searchQuery?.trim()) {
        params.append("search", searchQuery.trim());
      }

      const { data } = await api.get<any>(
        `/gcs/dynamic-destination-paths?path=${encodeURIComponent(
          path
        )}&${params.toString()}`
      );
      return data;
    },
    getNextPageParam: (lastPage) => {
      return lastPage.pagination?.has_next
        ? lastPage.pagination.current_page + 1
        : undefined;
    },
    initialPageParam: 1,
    enabled: !!path,
    placeholderData: keepPreviousData,
  });
}

// Get dynamic path configuration
export function useGetPathConfig() {
  return useQuery({
    queryKey: ["gcs", "path-config"],
    queryFn: async () => {
      const { data } = await api.get("/gcs/dynamic-path-config");
      return data;
    },
  });
}

// Update dynamic path configuration
export function useUpdatePathConfig() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (input: {
      base_path?: string;
      source_folder?: string;
    }) => {
      const { data } = await api.put("/gcs/dynamic-path-config", input);
      return data;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["gcs", "path-config"] });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-with-subfolders"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "destinations-with-subfolders"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-path"],
      });
      queryClient.invalidateQueries({
        queryKey: ["gcs", "list-destination-path"],
      });
      queryClient.invalidateQueries({
        queryKey: ["transferLogs"],
      });
      toast.success(data.message || "Path configuration updated successfully");
    },
    onError: (error: any) => {
      toast.error(
        error.response?.data?.message || "Failed to update path configuration"
      );
    },
  });
}

// Check file eligibility
export function useCheckFileEligibility(destinationPath: string) {
  return useQuery({
    queryKey: ["gcs", "check-eligibility", destinationPath],
    queryFn: async () => {
      if (!destinationPath) return null;
      const { data } = await api.post<FileEligibility>("/gcs/check-file-eligibility", {
        destination_path: destinationPath,
      });
      return data;
    },
    enabled: !!destinationPath,
  });
}

// Undo transfer
export function useUndoTransfer() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (transferId: string) => {
      const { data } = await api.post<UndoTransferResponse>("/gcs/undo-transfer", {
        transfer_id: transferId,
      });
      return data;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["gcs"] });
      queryClient.invalidateQueries({ queryKey: ["transferLogs"] });
      toast.success(data.message || "Transfer undone successfully");
    },
    onError: (error: any) => {
      toast.error(
        error.response?.data?.error ||
        error.response?.data?.message ||
        "Failed to undo transfer"
      );
    },
  });
}

// Delete destination file
export function useDeleteDestinationFile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (destinationPath: string) => {
      const { data } = await api.delete<DeleteDestinationFileResponse>("/gcs/delete-destination-file", {
        data: { destination_path: destinationPath },
      });
      return data;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["gcs"] });
      queryClient.invalidateQueries({ queryKey: ["transferLogs"] });
      toast.success(data.message || "File deleted successfully");
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.error ||
        error.response?.data?.message || "Failed to delete destination file"
      );
    },
  });
}
