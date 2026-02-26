import { useQuery } from "@tanstack/react-query";
import axiosInstance from "@/lib/axiosInstance";

interface GcsFile {
  name: string;
  size: number;
  updatedAt: string;
  contentType: string;
}

interface GcsFolder {
  name: string;
  display_name?: string; // Optional display name for cleaner UI
  updatedAt: string;
}

interface PaginationInfo {
  current_page: number;
  total_pages: number;
  page_size: number;
  total_items: number;
  total_folders: number;
  total_files: number;
  has_next: boolean;
  has_prev: boolean;
}

interface ApiResponse {
  folders: GcsFolder[];
  files: GcsFile[];
  pagination?: PaginationInfo;
}

const useFileBrowser = (url: string, enabled: boolean = true) => {
  const { data, isLoading, error, refetch, isFetching, isError, isSuccess } =
    useQuery({
      queryKey: ["fileBrowser", url],
      queryFn: async (): Promise<ApiResponse> => {
        const response = await axiosInstance.get<ApiResponse>(url);
        return response.data;
      },
      enabled: enabled && !!url, // Only fetch if enabled and url is provided
      staleTime: 5 * 60 * 1000, // Consider data fresh for 5 minutes
      gcTime: 10 * 60 * 1000, // Keep in cache for 10 minutes
      // retry: 3,
      // retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    });

  return {
    folders: data?.folders || [],
    files: data?.files || [],
    pagination: data?.pagination,
    loading: isLoading,
    error: error as Error | null,
    refetch,
    isFetching, // Shows if currently fetching (including background refetches)
    isError,
    isSuccess,
    // Additional useful functions
    invalidate: () => refetch(), // Alias for easier usage
  };
};

export default useFileBrowser;
