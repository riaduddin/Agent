import React from 'react';
import { Folder, File, FolderTree } from 'lucide-react';
import Pagination from './Pagination';
import EmptyPlaceholder from './EmptyPlaceholder';
import PreviewButton from './PreviewButton';

// Define types for files and folders
export interface GcsFile {
  name: string;
  size: number;
  updatedAt: string;
  contentType: string;
}

export interface GcsFolder {
  name: string;
  display_name?: string; // Optional display name for cleaner UI
  updatedAt: string;
}

interface PaginationInfo {
  current_page: number;
  total_pages: number;
  page_size: number;
  total_items: number;
  total_folders: number;
  total_files: number;
  has_next: boolean;
  has_prev: boolean;
}

interface FileBrowserProps {
  files: GcsFile[];
  folders: GcsFolder[];
  selectedFiles: GcsFile[];
  onFileSelect: (file: GcsFile) => void;
  onFolderNavigate: (folder: GcsFolder) => void;
  onFolderSelect?: (folder: GcsFolder) => void; // New prop for folder selection
  showFolderSelect?: boolean; // Whether to show folder select icons
  folderTransferDisabled?: boolean; // Whether folder transfers are disabled
  folderTransferDisabledMessage?: string; // Message to show when disabled
  pagination?: PaginationInfo;
  onPageChange?: (page: number) => void;
  onPageSizeChange?: (pageSize: number) => void;
  loading?: boolean;
  type?: 'source' | 'destination'; // For empty placeholder
  onFilePreview?: (filePath: string, fileName: string) => void; // NEW - optional preview function
}

const FileBrowser: React.FC<FileBrowserProps> = ({ 
  files = [], 
  folders = [], 
  selectedFiles, 
  onFileSelect, 
  onFolderNavigate,
  onFolderSelect,
  showFolderSelect = false,
  folderTransferDisabled = false,
  folderTransferDisabledMessage = "Please select a destination folder first",
  pagination,
  onPageChange,
  onPageSizeChange,
  loading = false,
  type = 'source',
  onFilePreview // NEW - optional preview function
}) => {
  const isSelected = (file: GcsFile) => {
    return selectedFiles.some((selectedFile) => selectedFile.name === file.name);
  };

  const handleFolderClick = (folder: GcsFolder, event: React.MouseEvent) => {
    // Prevent navigation when clicking the select button
    if ((event.target as HTMLElement).closest('.folder-select-btn')) {
      return;
    }
    onFolderNavigate(folder);
  };

  const handleFolderSelect = (folder: GcsFolder, event: React.MouseEvent) => {
    event.stopPropagation();
    if (folderTransferDisabled) {
      // Don't proceed if transfers are disabled
      return;
    }
    if (onFolderSelect) {
      onFolderSelect(folder);
    }
  };

  // Check if directory is empty
  const isEmpty = files.length === 0 && folders.length === 0;

  return (
    <div>
      {isEmpty ? (
        <EmptyPlaceholder type={type} />
      ) : (
        <div className="grid grid-cols-3 md:grid-cols-4 gap-4">
          {/* Render folders */}
          {folders.map((folder) => (
            <div 
              key={folder.name} 
              onClick={(e) => handleFolderClick(folder, e)} 
              className="cursor-pointer text-center p-2 hover:bg-gray-100 rounded-lg relative group"
            >
              <Folder className="w-12 h-12 mx-auto text-gray-500" />
              <p className="text-sm mt-1 truncate">
                {folder.display_name || folder.name.split('/').slice(-2).join('')}
              </p>
              
              {/* Folder Select Button - Only show in source panel */}
              {showFolderSelect && onFolderSelect && (
                <button
                  onClick={(e) => handleFolderSelect(folder, e)}
                  className={`folder-select-btn absolute top-1 right-1 p-1 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-200 ${
                    folderTransferDisabled 
                      ? 'bg-gray-400 text-gray-200 cursor-not-allowed' 
                      : 'bg-blue-500 text-white hover:bg-blue-600 cursor-pointer'
                  }`}
                  title={folderTransferDisabled ? folderTransferDisabledMessage : "Transfer entire folder"}
                  disabled={folderTransferDisabled}
                >
                  <FolderTree className="w-4 h-4" />
                </button>
              )}
            </div>
          ))}
          
          {/* Render files */}
          {files.map((file) => (
            <div
              key={file.name}
              onClick={() => onFileSelect(file)}
              className={`cursor-pointer text-center p-2 hover:bg-gray-100 rounded-lg relative group ${isSelected(file) ? 'bg-blue-100' : ''}`}
            >
              <File className="w-12 h-12 mx-auto text-gray-500" />
              <p className="text-sm mt-1 truncate">{file.name.split('/').pop()}</p>
              
              {/* Preview Button - Only show if onFilePreview is provided */}
              {onFilePreview && (
                <div className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                  <PreviewButton 
                    file={file} 
                    onPreview={onFilePreview} 
                  />
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    
      {/* Pagination - only show if not empty */}
      {!isEmpty && pagination && onPageChange && onPageSizeChange && (
        <Pagination
          currentPage={pagination.current_page}
          totalPages={pagination.total_pages}
          pageSize={pagination.page_size}
          totalItems={pagination.total_items}
          totalFolders={pagination.total_folders}
          totalFiles={pagination.total_files}
          hasNext={pagination.has_next}
          hasPrev={pagination.has_prev}
          onPageChange={onPageChange}
          onPageSizeChange={onPageSizeChange}
          loading={loading}
        />
      )}
    </div>
  );
};

export default FileBrowser;
