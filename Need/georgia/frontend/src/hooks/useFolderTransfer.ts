/* eslint-disable @typescript-eslint/no-explicit-any */
import { useState } from "react";
import axiosInstance from "@/lib/axiosInstance";
import { toast } from "sonner";

interface FolderTransferRequest {
  sourceFolderPath: string;
  destination: {
    root: string;
    path: string;
  };
  operation: "move" | "copy";
}

interface FolderTransferResponse {
  transferId: string;
  message: string;
  transferred_files: Array<{
    source: string;
    destination: string;
    operation: string;
  }>;
  total_files: number;
  errors: string[];
  status: "succeeded" | "partial";
}

export const useFolderTransfer = (onSuccess?: () => void) => {
  const [isTransferring, setIsTransferring] = useState(false);

  const transferFolder = async (
    request: FolderTransferRequest
  ): Promise<FolderTransferResponse | null> => {
    setIsTransferring(true);
    const toastId = toast.loading(
      `${request.operation === "move" ? "Moving" : "Copying"} folder...`
    );

    try {
      const response = await axiosInstance.post<FolderTransferResponse>(
        "/gcs/transfer-folder",
        request
      );

      const result = response.data;

      if (result.status === "succeeded") {
        toast.success(
          `Successfully ${request.operation}d folder! ${result.total_files} files transferred.`,
          { id: toastId }
        );
      } else if (result.status === "partial") {
        toast.warning(
          `Folder ${request.operation}d with some errors. ${result.total_files} files transferred, ${result.errors.length} errors.`,
          { id: toastId }
        );
      }

      // Always call onSuccess for successful transfers to refresh UI
      if (onSuccess) {
        onSuccess();
      }

      return result;
    } catch (error: any) {
      console.error("Folder transfer error:", error);
      toast.error(
        `Error ${request.operation === "move" ? "moving" : "copying"} folder: ${
          error.response?.data?.message || error.message
        }`,
        { id: toastId }
      );
      return null;
    } finally {
      setIsTransferring(false);
    }
  };

  return {
    transferFolder,
    isTransferring,
  };
};

export default useFolderTransfer;
