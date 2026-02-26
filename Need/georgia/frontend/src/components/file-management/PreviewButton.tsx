import React from 'react';
import { Button } from '@/components/ui/button';
import { Eye } from 'lucide-react';
import { GcsFile } from './FileBrowser';

interface PreviewButtonProps {
  file: GcsFile;
  onPreview: (filePath: string, fileName: string) => void;
}

const PreviewButton: React.FC<PreviewButtonProps> = ({ file, onPreview }) => {
  const handlePreviewClick = (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent any parent click handlers
    onPreview(file.name, file.name); // Use file.name for both path and display name
  };

  // Only show preview button for PDF files
  const isPdf = file.name.toLowerCase().endsWith('.pdf');
  
  if (!isPdf) {
    return null;
  }

  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={handlePreviewClick}
      title={`Preview ${file.name}`}
      className="h-8 w-8 p-0 hover:bg-gray-100"
    >
      <Eye className="h-4 w-4 text-gray-600" />
    </Button>
  );
};

export default PreviewButton;
