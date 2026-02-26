import React, { useEffect } from "react";
import FileBrowser, { GcsFolder, GcsFile } from "./FileBrowser";
import Breadcrumbs from "./Breadcrumbs";
import FileBrowserSkeleton from "./FileBrowserSkeleton";

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

interface DestinationPanelProps {
  currentPath: string;
  setCurrentPath: (path: string) => void;
  folders: GcsFolder[];
  files: GcsFile[];
  loading: boolean;
  error: Error | null;
  onPathChange: (path: string) => void; // for transfers
  pagination?: PaginationInfo;
  onPageChange?: (page: number) => void;
  onPageSizeChange?: (pageSize: number) => void;
}

const DestinationPanel: React.FC<DestinationPanelProps> = ({
  currentPath,
  setCurrentPath,
  folders,
  files,
  loading,
  error,
  onPathChange,
  pagination,
  onPageChange,
  onPageSizeChange,
}) => {
  useEffect(() => {
    onPathChange(currentPath);
  }, [currentPath, onPathChange]);

  const handleFolderNavigate = (folder: GcsFolder) => {
    setCurrentPath(folder.name); // update path
  };

  const handleBreadcrumbNavigate = (path: string) => {
    setCurrentPath(path); // navigate via breadcrumbs
  };

  const getBreadcrumbs = () => {
    const crumbs = [{ name: "Georgia 14", path: "" }];
    if (currentPath) {
      const parts = currentPath.split("/").filter(Boolean);

      let path = "";

      for (const part of parts) {
        path += `${part}/`;

        // Skip if this breadcrumb name is the same as the previous one (avoid duplicates)
        const lastCrumb = crumbs[crumbs.length - 1];

        if (lastCrumb && lastCrumb.name === part) {
          continue; // Skip duplicate
        }

        crumbs.push({ name: part, path });
      }
    }
    return crumbs;
  };

  return (
    <div className="h-full">
      <div className="mb-4">
        <Breadcrumbs
          items={getBreadcrumbs()}
          onNavigate={handleBreadcrumbNavigate}
        />
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
          selectedFiles={[]} // destination doesn't select files
          onFileSelect={() => { }}
          onFolderNavigate={handleFolderNavigate}
          pagination={pagination}
          onPageChange={onPageChange}
          onPageSizeChange={onPageSizeChange}
          loading={loading}
          type="destination"
        />
      )}
    </div>
  );
};

export default DestinationPanel;
