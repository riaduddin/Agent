"use client";

import React, { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { FolderPlus, Folder } from 'lucide-react';

interface CreateFolderDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (folderName: string) => void;
  currentPath: string;
  isCreating: boolean;
}

const CreateFolderDialog: React.FC<CreateFolderDialogProps> = ({
  isOpen,
  onClose,
  onConfirm,
  currentPath,
  isCreating
}) => {
  const [folderName, setFolderName] = useState('');
  const [error, setError] = useState('');

  const handleClose = () => {
    setFolderName('');
    setError('');
    onClose();
  };

  const handleConfirm = () => {
    const trimmedName = folderName.trim();
    
    if (!trimmedName) {
      setError('Folder name is required');
      return;
    }

    if (trimmedName.includes('/') || trimmedName.includes('\\')) {
      setError('Folder name cannot contain slashes');
      return;
    }

    if (trimmedName.length > 100) {
      setError('Folder name is too long (max 100 characters)');
      return;
    }

    // Check for invalid characters
    const invalidChars = /[<>:"|?*]/;
    if (invalidChars.test(trimmedName)) {
      setError('Folder name contains invalid characters');
      return;
    }

    setError('');
    onConfirm(trimmedName);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleConfirm();
    }
  };

  const getCurrentPathDisplay = () => {
    if (!currentPath) return 'Root';
    const parts = currentPath.split('/').filter(Boolean);
    return parts[parts.length - 1] || 'Root';
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FolderPlus className="w-5 h-5 text-green-500" />
            Create New Folder
          </DialogTitle>
          <DialogDescription>
            Create a new folder in the current directory
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="bg-gray-50 p-3 rounded-lg">
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <Folder className="w-4 h-4" />
              <span>Location: <strong>{getCurrentPathDisplay()}</strong></span>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="folderName">Folder Name</Label>
            <Input
              id="folderName"
              value={folderName}
              onChange={(e) => setFolderName(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Enter folder name..."
              disabled={isCreating}
              className={error ? 'border-red-500' : ''}
            />
            {error && (
              <p className="text-sm text-red-500">{error}</p>
            )}
          </div>

          <div className="text-xs text-gray-500 bg-blue-50 p-3 rounded border border-blue-200">
            <strong>💡 Tip:</strong> Folder names should be descriptive and cannot contain special characters like / \ : " | ? * &lt; &gt;
          </div>
        </div>

        <DialogFooter>
          <Button 
            variant="outline" 
            onClick={handleClose}
            disabled={isCreating}
          >
            Cancel
          </Button>
          <Button 
            onClick={handleConfirm}
            disabled={isCreating || !folderName.trim()}
            className="bg-green-600 hover:bg-green-700"
          >
            {isCreating ? "Creating..." : "Create Folder"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default CreateFolderDialog;