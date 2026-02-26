"use client";

import React from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { X, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';

export interface TransferProgress {
  id: string;
  type: 'file' | 'folder';
  operation: 'copy' | 'move';
  totalItems: number;
  completedItems: number;
  currentItem?: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  error?: string;
  startTime: Date;
  endTime?: Date;
}

interface TransferProgressDialogProps {
  isOpen: boolean;
  onClose: () => void;
  transfers: TransferProgress[];
  onCancel?: (transferId: string) => void;
}

const TransferProgressDialog: React.FC<TransferProgressDialogProps> = ({
  isOpen,
  onClose,
  transfers,
  onCancel
}) => {
  const activeTransfers = transfers.filter(t => t.status === 'in_progress' || t.status === 'pending');
  const completedTransfers = transfers.filter(t => t.status === 'completed');
  const failedTransfers = transfers.filter(t => t.status === 'failed');

  const getProgressPercentage = (transfer: TransferProgress) => {
    if (transfer.totalItems === 0) return 0;
    return Math.round((transfer.completedItems / transfer.totalItems) * 100);
  };

  const getStatusIcon = (status: TransferProgress['status']) => {
    switch (status) {
      case 'pending':
        return <Loader2 className="h-4 w-4 animate-spin text-blue-500" />;
      case 'in_progress':
        return <Loader2 className="h-4 w-4 animate-spin text-blue-500" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return null;
    }
  };

  const formatDuration = (start: Date, end?: Date) => {
    const endTime = end || new Date();
    const duration = Math.round((endTime.getTime() - start.getTime()) / 1000);
    
    if (duration < 60) return `${duration}s`;
    if (duration < 3600) return `${Math.floor(duration / 60)}m ${duration % 60}s`;
    return `${Math.floor(duration / 3600)}h ${Math.floor((duration % 3600) / 60)}m`;
  };

  const canClose = activeTransfers.length === 0;

  return (
    <Dialog open={isOpen} onOpenChange={canClose ? onClose : undefined}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader className="flex flex-row items-center justify-between">
          <DialogTitle className="text-lg font-semibold">
            Transfer Progress
          </DialogTitle>
          {canClose && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="h-6 w-6 p-0"
            >
              <X className="h-4 w-4" />
            </Button>
          )}
        </DialogHeader>

        <div className="space-y-4">
          {/* Summary */}
          <div className="bg-gray-50 p-3 rounded-lg">
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div className="text-center">
                <div className="font-medium text-blue-600">{activeTransfers.length}</div>
                <div className="text-gray-500">Active</div>
              </div>
              <div className="text-center">
                <div className="font-medium text-green-600">{completedTransfers.length}</div>
                <div className="text-gray-500">Completed</div>
              </div>
              <div className="text-center">
                <div className="font-medium text-red-600">{failedTransfers.length}</div>
                <div className="text-gray-500">Failed</div>
              </div>
            </div>
          </div>

          {/* Active Transfers */}
          {activeTransfers.length > 0 && (
            <div>
              <h3 className="font-medium text-gray-900 mb-2">Active Transfers</h3>
              <div className="space-y-3">
                {activeTransfers.map((transfer) => (
                  <div key={transfer.id} className="border border-gray-200 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {getStatusIcon(transfer.status)}
                        <span className="font-medium text-sm">
                          {transfer.operation === 'move' ? 'Moving' : 'Copying'} {transfer.type}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-500">
                          {transfer.completedItems}/{transfer.totalItems} items
                        </span>
                        {onCancel && transfer.status === 'in_progress' && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => onCancel(transfer.id)}
                            className="h-6 px-2 text-xs"
                          >
                            Cancel
                          </Button>
                        )}
                      </div>
                    </div>
                    
                    <Progress 
                      value={getProgressPercentage(transfer)} 
                      className="h-2 mb-2" 
                    />
                    
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>{getProgressPercentage(transfer)}% complete</span>
                      <span>{formatDuration(transfer.startTime)}</span>
                    </div>
                    
                    {transfer.currentItem && (
                      <div className="text-xs text-gray-600 mt-1 truncate">
                        Current: {transfer.currentItem}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Completed Transfers */}
          {completedTransfers.length > 0 && (
            <div>
              <h3 className="font-medium text-gray-900 mb-2">Completed</h3>
              <div className="space-y-2">
                {completedTransfers.slice(-3).map((transfer) => (
                  <div key={transfer.id} className="flex items-center justify-between p-2 bg-green-50 rounded">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(transfer.status)}
                      <span className="text-sm">
                        {transfer.operation === 'move' ? 'Moved' : 'Copied'} {transfer.totalItems} items
                      </span>
                    </div>
                    <span className="text-xs text-gray-500">
                      {formatDuration(transfer.startTime, transfer.endTime)}
                    </span>
                  </div>
                ))}
                {completedTransfers.length > 3 && (
                  <div className="text-xs text-gray-500 text-center">
                    +{completedTransfers.length - 3} more completed
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Failed Transfers */}
          {failedTransfers.length > 0 && (
            <div>
              <h3 className="font-medium text-gray-900 mb-2">Failed</h3>
              <div className="space-y-2">
                {failedTransfers.map((transfer) => (
                  <div key={transfer.id} className="p-2 bg-red-50 rounded">
                    <div className="flex items-center gap-2 mb-1">
                      {getStatusIcon(transfer.status)}
                      <span className="text-sm">
                        Failed to {transfer.operation} {transfer.type}
                      </span>
                    </div>
                    {transfer.error && (
                      <div className="text-xs text-red-600 ml-6">
                        {transfer.error}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* No transfers */}
          {transfers.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              No transfers in progress
            </div>
          )}
        </div>

        {/* Footer */}
        {canClose && (
          <div className="flex justify-end pt-4 border-t">
            <Button onClick={onClose} variant="outline">
              Close
            </Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default TransferProgressDialog;
