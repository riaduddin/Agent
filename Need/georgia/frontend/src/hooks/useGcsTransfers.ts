import { useState, useEffect } from 'react';
import axiosInstance from '@/lib/axiosInstance';
import { Transfer } from '@/types/transfer';

const useGcsTransfers = (
  invalidateSourceFiles: () => void,
  refetchDestination: () => void,
) => {
  const [transfers, setTransfers] = useState<Transfer[]>([]);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const initiateTransfer = async (
    sourcePath: string,
    destinationPath: string,
    operation: "move" | "copy"
  ) => {
    setIsModalOpen(true);
    try {
      const response = await axiosInstance.post("/gcs/transfer", {
        source: { path: sourcePath },
        destination: { root: destinationPath, path: "" },
        operation,
      });

      const newTransfer: Transfer = {
        id: response.data.transferId,
        fileName: sourcePath.split("/").pop() || "unknown",
        status: "running", // Start as running
        progress: 0,
        source: sourcePath,
        destination: destinationPath,
        timestamp: new Date().toISOString(),
      };

      setTransfers((prev) => [newTransfer, ...prev]);
      setIsModalOpen(false);
      invalidateSourceFiles();
      refetchDestination();
    } catch (error) {
      console.error("Failed to initiate transfer", error);
      setIsModalOpen(false);
    }
  };

  const fetchTransfers = async () => {
    try {
      const response = await axiosInstance.get("/gcs/transfers");
      setTransfers(response.data);
    } catch (error) {
      console.error("Failed to fetch transfers", error);
    }
  };

  useEffect(() => {
    fetchTransfers();
  }, []);

  // Polling for transfer status
  useEffect(() => {
    const interval = setInterval(() => {
      transfers.forEach(async (transfer) => {
        if (transfer.status === "running" || transfer.status === "queued") {
          const response = await axiosInstance.get(
            `/gcs/transfers/${transfer.id}`
          );
          setTransfers((prev) =>
            prev.map((t) =>
              t.id === transfer.id ? { ...t, ...response.data } : t
            )
          );
        }
      });
    }, 5000); // Poll every 5 seconds

    return () => clearInterval(interval);
  }, [transfers]);

  return {
    transfers,
    initiateTransfer,
    fetchTransfers,
    isModalOpen,
    setIsModalOpen,
  };
};

export default useGcsTransfers;
