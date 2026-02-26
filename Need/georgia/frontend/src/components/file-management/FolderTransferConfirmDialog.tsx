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
import { FolderTree, ArrowRight } from 'lucide-react';

interface FolderTransferConfirmDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  folderName: string;
  destinationPath: string;
  isTransferring: boolean;
}

const FolderTransferConfirmDialog: React.FC<FolderTransferConfirmDialogProps> = ({
  isOpen,
  onClose,
  onConfirm,
  folderName,
  destinationPath,
  isTransferring
}) => {
  const getFolderDisplayName = (path: string) => {
    return path.split('/').filter(Boolean).pop() || path;
  };

  const getDestinationDisplayName = (path: string) => {
    return path.split('/').filter(Boolean).pop() || 'Root';
  };

  return (
    <AlertDialog open={isOpen} onOpenChange={onClose}>
      <AlertDialogContent className="max-w-md">
        <AlertDialogHeader>
          <AlertDialogTitle className="flex items-center gap-2">
            <FolderTree className="w-5 h-5 text-blue-500" />
            Confirm Folder Transfer
          </AlertDialogTitle>
          <AlertDialogDescription className="space-y-4">
            <p>
              Are you sure you want to move the entire folder and all its contents?
            </p>
            
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 bg-blue-100 rounded flex items-center justify-center">
                    <FolderTree className="w-4 h-4 text-blue-600" />
                  </div>
                  <span className="font-medium text-gray-900">
                    {getFolderDisplayName(folderName)}
                  </span>
                </div>
                
                <ArrowRight className="w-4 h-4 text-gray-400" />
                
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 bg-green-100 rounded flex items-center justify-center">
                    <FolderTree className="w-4 h-4 text-green-600" />
                  </div>
                  <span className="font-medium text-gray-900">
                    {getDestinationDisplayName(destinationPath)}
                  </span>
                </div>
              </div>
            </div>

            <div className="text-xs text-gray-500 bg-yellow-50 p-3 rounded border border-yellow-200">
              <strong>⚠️ Warning:</strong> This will move the folder and all files/subfolders inside it. 
              The original folder will be removed from the source location.
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
            {isTransferring ? "Moving..." : "Move Folder"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
};

export default FolderTransferConfirmDialog;