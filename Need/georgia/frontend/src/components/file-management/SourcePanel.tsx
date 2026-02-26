import React, { useState } from 'react';
import FileBrowser, { GcsFile, GcsFolder } from './FileBrowser';
import Breadcrumbs from './Breadcrumbs';
import FileBrowserSkeleton from './FileBrowserSkeleton';
import FolderTransferConfirmDialog from './FolderTransferConfirmDialog';
import CreateFolderDialog from './CreateFolderDialog';
import { Button } from '@/components/ui/button';
import { FolderPlus, Upload, ArrowRight } from 'lucide-react';
import useCreateFolder from '@/hooks/useCreateFolder';

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

interface SourcePanelProps {
  selectedFiles: GcsFile[];
  onFileSelect: (file: GcsFile) => void;
  currentPath: string;
  setCurrentPath: (path: string) => void;
  sourceRoot: string; // Add sourceRoot prop
  onFolderTransfer?: (folder: GcsFolder) => void; // New prop for folder transfer
  destinationPath?: string; // For confirmation dialog
  isTransferring?: boolean; // For confirmation dialog
  onRefresh?: () => void; // For refreshing after folder creation

  folders: GcsFolder[];
  files: GcsFile[];
  loading: boolean;
  error: Error | null;
  pagination?: PaginationInfo;
  onPageChange?: (page: number) => void;
  onPageSizeChange?: (pageSize: number) => void;

  // New props for action buttons
  onUploadClick?: () => void;
  onTransferSelected?: () => void;
  fileUploading?: boolean;
  onFilePreview?: (filePath: string, fileName: string) => void; // NEW - preview function
}

const SourcePanel: React.FC<SourcePanelProps> = ({
  selectedFiles,
  onFileSelect,
  currentPath,
  setCurrentPath,
  sourceRoot,
  onFolderTransfer,
  destinationPath,
  isTransferring = false,
  onRefresh,
  folders,
  files,
  loading,
  error,
  pagination,
  onPageChange,
  onPageSizeChange,
  // New props
  onUploadClick,
  onTransferSelected,
  fileUploading = false,
  onFilePreview // NEW - preview function
}) => {
  const [showTransferConfirm, setShowTransferConfirm] = useState(false);
  const [showCreateFolder, setShowCreateFolder] = useState(false);
  const [selectedFolder, setSelectedFolder] = useState<GcsFolder | null>(null);

  const { createFolder, isCreating } = useCreateFolder(() => {
    if (onRefresh) {
      onRefresh();
    }
  });
  const handleFolderNavigate = (folder: GcsFolder) => {
    setCurrentPath(folder.name);
  };

  const handleBreadcrumbNavigate = (path: string) => {
    if (path === "") {
      // Navigate to source root (this is the highest level allowed)
      setCurrentPath(sourceRoot);
    } else {
      // Only allow navigation within or below the source root
      if (isPathWithinSourceRoot(path)) {
        setCurrentPath(path);
      }
    }
  };

  // Helper function to check if a path is within the source root
  const isPathWithinSourceRoot = (path: string) => {
    if (!sourceRoot || !path) return false;

    // Normalize paths by removing trailing slashes for comparison
    const normalizedSourceRoot = sourceRoot.replace(/\/$/, '');
    const normalizedPath = path.replace(/\/$/, '');

    // Path must start with source root or be equal to it
    return normalizedPath === normalizedSourceRoot || normalizedPath.startsWith(normalizedSourceRoot + '/');
  };

  // Helper function to get the display name for source root
  const getSourceRootDisplayName = () => {
    if (!sourceRoot) return "Source";

    const parts = sourceRoot.split("/").filter(Boolean);
    return parts[parts.length - 1] || "Source"; // Get the last part as display name
  };

  const handleFolderTransferRequest = (folder: GcsFolder) => {
    if (!destinationPath) {
      // This should now be prevented by the disabled button, but keep as fallback
      return;
    }
    setSelectedFolder(folder);
    setShowTransferConfirm(true);
  };

  const handleConfirmTransfer = () => {
    if (selectedFolder && onFolderTransfer) {
      onFolderTransfer(selectedFolder);
    }
    setShowTransferConfirm(false);
    setSelectedFolder(null);
  };

  const handleCreateFolder = async (folderName: string) => {
    await createFolder({
      folderName,
      currentPath
    });
    setShowCreateFolder(false);
  };

  const getBreadcrumbs = () => {
    if (!sourceRoot || !currentPath) {
      return [{ name: "Source", path: "" }];
    }

    const sourceRootParts = sourceRoot.split("/").filter(Boolean);
    const currentParts = currentPath.split("/").filter(Boolean);

    // Start with the source root as the base breadcrumb with a visual indicator
    const sourceDisplayName = getSourceRootDisplayName();
    const crumbs = [{
      name: `🏠 ${sourceDisplayName}`, // Add home icon to indicate this is the root boundary
      path: sourceRoot
    }];

    // Only show parts that are deeper than the source root
    if (currentParts.length > sourceRootParts.length) {
      const relativeParts = currentParts.slice(sourceRootParts.length);
      let path = sourceRoot;

      for (const part of relativeParts) {
        path += `${part}/`;
        crumbs.push({ name: part, path });
      }
    }

    return crumbs;
  };

  // Check if we're at the source root boundary
  const isAtSourceRoot = currentPath === sourceRoot;

  return (
    <div className="h-full">
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex-1">
            <Breadcrumbs
              items={getBreadcrumbs()}
              onNavigate={handleBreadcrumbNavigate}
            />
          </div>

          {/* Action Buttons Group */}
          <div className="flex items-center gap-2 ml-2">
            {onUploadClick && (
              <Button
                onClick={onUploadClick}
                size="sm"
                variant="outline"
                className="text-blue-600 border-blue-300 hover:bg-blue-50"
                disabled={loading || fileUploading}
              >
                <Upload className="w-4 h-4 mr-1" />
                {fileUploading ? "Uploading..." : "Upload Files"}
              </Button>
            )}

            {onTransferSelected && (
              <Button
                onClick={onTransferSelected}
                size="sm"
                className="bg-blue-500 hover:bg-blue-600 text-white"
                disabled={selectedFiles.length === 0 || !destinationPath || isTransferring || loading}
              >
                <ArrowRight className="w-4 h-4 mr-1" />
                {isTransferring ? "Transferring..." : `Transfer (${selectedFiles.length})`}
              </Button>
            )}

            <Button
              onClick={() => setShowCreateFolder(true)}
              size="sm"
              variant="outline"
              className="text-green-600 border-green-300 hover:bg-green-50"
              disabled={loading || isCreating}
            >
              <FolderPlus className="w-4 h-4 mr-1" />
              Add Folder
            </Button>
          </div>
        </div>
      </div>

      {loading && <FileBrowserSkeleton />}
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-600 font-medium">Error: {error.message}</p>
        </div>
      )}
      {!loading && !error && (
        <FileBrowser
          folders={folders}
          files={files}
          selectedFiles={selectedFiles}
          onFileSelect={onFileSelect}
          onFolderNavigate={handleFolderNavigate}
          onFolderSelect={handleFolderTransferRequest}
          showFolderSelect={true}
          folderTransferDisabled={!destinationPath}
          folderTransferDisabledMessage="Please select a destination directory first. Navigate to a specific folder in the destination panel."
          pagination={pagination}
          onPageChange={onPageChange}
          onPageSizeChange={onPageSizeChange}
          loading={loading}
          type="source"
          onFilePreview={onFilePreview}
        />
      )}

      {/* Folder Transfer Confirmation Dialog */}
      <FolderTransferConfirmDialog
        isOpen={showTransferConfirm}
        onClose={() => {
          setShowTransferConfirm(false);
          setSelectedFolder(null);
        }}
        onConfirm={handleConfirmTransfer}
        folderName={selectedFolder?.name || ''}
        destinationPath={destinationPath || ''}
        isTransferring={isTransferring}
      />

      {/* Create Folder Dialog */}
      <CreateFolderDialog
        isOpen={showCreateFolder}
        onClose={() => setShowCreateFolder(false)}
        onConfirm={handleCreateFolder}
        currentPath={currentPath}
        isCreating={isCreating}
      />
    </div>
  );
};

export default SourcePanel;
