import { useState, useCallback, useRef } from 'react';
import { TransferProgress } from '@/components/file-management/TransferProgressDialog';

export const useTransferProgress = () => {
  const [transfers, setTransfers] = useState<TransferProgress[]>([]);
  const [isProgressDialogOpen, setIsProgressDialogOpen] = useState(false);
  const transferIdCounter = useRef(0);

  const createTransfer = useCallback((
    type: 'file' | 'folder',
    operation: 'copy' | 'move',
    totalItems: number
  ): string => {
    const id = `transfer-${++transferIdCounter.current}`;
    const newTransfer: TransferProgress = {
      id,
      type,
      operation,
      totalItems,
      completedItems: 0,
      status: 'pending',
      startTime: new Date()
    };

    setTransfers(prev => [...prev, newTransfer]);
    setIsProgressDialogOpen(true);
    
    return id;
  }, []);

  const updateTransfer = useCallback((
    id: string, 
    updates: Partial<TransferProgress>
  ) => {
    setTransfers(prev => prev.map(transfer => 
      transfer.id === id 
        ? { 
            ...transfer, 
            ...updates,
            endTime: updates.status === 'completed' || updates.status === 'failed' 
              ? new Date() 
              : transfer.endTime
          }
        : transfer
    ));
  }, []);

  const startTransfer = useCallback((id: string) => {
    updateTransfer(id, { status: 'in_progress' });
  }, [updateTransfer]);

  const updateProgress = useCallback((
    id: string, 
    completedItems: number, 
    currentItem?: string
  ) => {
    updateTransfer(id, { completedItems, currentItem });
  }, [updateTransfer]);

  const completeTransfer = useCallback((id: string) => {
    updateTransfer(id, { status: 'completed' });
  }, [updateTransfer]);

  const failTransfer = useCallback((id: string, error: string) => {
    updateTransfer(id, { status: 'failed', error });
  }, [updateTransfer]);

  const cancelTransfer = useCallback((id: string) => {
    // In a real implementation, this would send a cancel request to the backend
    updateTransfer(id, { status: 'failed', error: 'Transfer cancelled by user' });
  }, [updateTransfer]);

  const clearCompletedTransfers = useCallback(() => {
    setTransfers(prev => prev.filter(t => t.status !== 'completed'));
  }, []);

  const closeProgressDialog = useCallback(() => {
    const hasActiveTransfers = transfers.some(t => 
      t.status === 'pending' || t.status === 'in_progress'
    );
    
    if (!hasActiveTransfers) {
      setIsProgressDialogOpen(false);
    }
  }, [transfers]);

  const openProgressDialog = useCallback(() => {
    setIsProgressDialogOpen(true);
  }, []);

  return {
    transfers,
    isProgressDialogOpen,
    createTransfer,
    startTransfer,
    updateProgress,
    completeTransfer,
    failTransfer,
    cancelTransfer,
    clearCompletedTransfers,
    closeProgressDialog,
    openProgressDialog
  };
};
