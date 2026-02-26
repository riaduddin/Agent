import React from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";

import { Transfer } from '@/types/transfer';

interface TransferProgressModalProps {
  isOpen: boolean;
  onClose: () => void;
  transfers: Transfer[];
}

const TransferProgressModal: React.FC<TransferProgressModalProps> = ({ isOpen, onClose, transfers }) => {
  const inProgressTransfers = transfers.filter(t => t.status === 'running' || t.status === 'queued');
  const totalProgress = inProgressTransfers.reduce((acc, t) => acc + t.progress, 0) / (inProgressTransfers.length || 1);

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Transferring Files</DialogTitle>
          <DialogDescription>
            Please wait while your files are being transferred.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <p className="mb-2">Overall Progress</p>
            <Progress value={totalProgress} />
          </div>
          <div className="space-y-2">
            {inProgressTransfers.map((transfer) => (
              <div key={transfer.id}>
                <p className="text-sm truncate">{transfer.fileName}</p>
                <Progress value={transfer.progress} />
              </div>
            ))}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default TransferProgressModal;
