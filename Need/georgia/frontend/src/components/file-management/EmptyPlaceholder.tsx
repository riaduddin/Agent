"use client";

import React from 'react';
import { FolderOpen, FileX } from 'lucide-react';

interface EmptyPlaceholderProps {
  type: 'source' | 'destination';
  message?: string;
}

const EmptyPlaceholder: React.FC<EmptyPlaceholderProps> = ({ 
  type, 
  message 
}) => {
  const defaultMessage = type === 'source' 
    ? 'No files or folders in this directory'
    : 'This destination folder is empty';

  return (
    <div className="flex flex-col items-center justify-center py-12 px-4 text-gray-500">
      <div className="mb-4">
        {type === 'source' ? (
          <FolderOpen className="h-16 w-16 text-gray-300" />
        ) : (
          <FileX className="h-16 w-16 text-gray-300" />
        )}
      </div>
      
      <h3 className="text-lg font-medium text-gray-600 mb-2">
        {type === 'source' ? 'Empty Directory' : 'Empty Destination'}
      </h3>
      
      <p className="text-sm text-gray-500 text-center max-w-sm">
        {message || defaultMessage}
      </p>
      
      {type === 'source' && (
        <p className="text-xs text-gray-400 mt-2 text-center">
          Use the "Add Folder" button to create new folders
        </p>
      )}
    </div>
  );
};

export default EmptyPlaceholder;
