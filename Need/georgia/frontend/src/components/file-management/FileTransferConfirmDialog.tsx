"use client";

import React from 'react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { FileText, ArrowRight } from 'lucide-react';

interface FileTransferConfirmDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  fileCount: number;
  destinationPath: string;
  isTransferring: boolean;
}

const FileTransferConfirmDialog: React.FC<FileTransferConfirmDialogProps> = ({
  isOpen,
  onClose,
  onConfirm,
  fileCount,
  destinationPath,
  isTransferring
}) => {
  const getDestinationDisplayName = (path: string) => {
    return path.split('/').filter(Boolean).pop() || 'Root';
  };

  return (
    <AlertDialog open={isOpen} onOpenChange={onClose}>
      <AlertDialogContent className="max-w-md">
        <AlertDialogHeader>
          <AlertDialogTitle className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-blue-500" />
            Confirm File Transfer
          </AlertDialogTitle>
          <AlertDialogDescription className="space-y-4">
            <p>
              Are you sure you want to move {fileCount} selected file{fileCount > 1 ? 's' : ''}?
            </p>
            
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 bg-blue-100 rounded flex items-center justify-center">
                    <FileText className="w-4 h-4 text-blue-600" />
                  </div>
                  <span className="font-medium text-gray-900">
                    {fileCount} file{fileCount > 1 ? 's' : ''}
                  </span>
                </div>
                
                <ArrowRight className="w-4 h-4 text-gray-400" />
                
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 bg-green-100 rounded flex items-center justify-center">
                    <FileText className="w-4 h-4 text-green-600" />
                  </div>
                  <span className="font-medium text-gray-900">
                    {getDestinationDisplayName(destinationPath)}
                  </span>
                </div>
              </div>
            </div>

            <div className="text-xs text-gray-500 bg-yellow-50 p-3 rounded border border-yellow-200">
              <strong>⚠️ Warning:</strong> This will move the selected files to the destination folder. 
              The original files will be removed from the source location.
            </div>
          </AlertDialogDescription>
        </AlertDialogHeader>
        
        <AlertDialogFooter>
          <AlertDialogCancel disabled={isTransferring}>
            Cancel
          </AlertDialogCancel>
          <AlertDialogAction 
            onClick={onConfirm}
            disabled={isTransferring}
            className="bg-blue-600 hover:bg-blue-700"
          >
            {isTransferring ? "Moving..." : `Move ${fileCount} File${fileCount > 1 ? 's' : ''}`}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
};

export default FileTransferConfirmDialog;
