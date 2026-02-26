/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import TreeView, {
  TreeViewItem,
  TreeViewMenuItem,
} from "@/components/ui/tree-view";
import { useState, useCallback, useMemo, useEffect } from "react";
import {
  AlertCircle,
  Check,
  Download,
  Edit,
  Eye,
  File,
  Folder,
  FolderOpen,
  FolderPlus,
  Globe,
  Info,
  Loader2,
  MoreVertical,
  Move,
  Search,
  Share2,
  Trash2,
  Undo2,
  Upload,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogDescription,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "sonner";
import {
  useCreateFolder,
  useUploadFilesWithConflict, // Use the new hook
  useResolveUploadConflict,   // Use the new hook
  useTransferFile,
  useTransferFolder,
  useTransferProgress,
  useDeleteGCS,
  useFilePreview,
  useRenameFolder,
  useRenameFile,
  useGCSListPathInfinite,
  useGCSListDestinationPathInfinite,
  useGetPathConfig,
  useUpdatePathConfig,
  useDeleteDestinationFile,
  useUndoTransfer, // Add new hook
  useCheckTransferConflicts,
} from "@/hooks/use-gcs";
import { Progress } from "@/components/ui/progress";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useDebounce } from "@/hooks/useDebounce";
import axios from "axios";
import { useAuth } from "@/context/AuthContext";
import { useFilePermissions } from "@/hooks/use-file-permissions";
import {
  createCategoryRegexPatterns,
  filterTreeDataByCategories,
} from "@/lib/gcs-formatter";
import { Badge } from "@/components/ui/badge"; // Import Badge if available, or just use div

// Helper function to merge new items into the tree
const mergeTreeData = (
  currentTree: TreeViewItem[],
  targetPath: string,
  newItems: TreeViewItem[]
): TreeViewItem[] => {
  return currentTree.map((item) => {
    // If this is the target folder, update its children
    if (item.id === targetPath) {
      return { ...item, children: newItems };
    }

    // If this item has children, recurse
    if (item.children) {
      return {
        ...item,
        children: mergeTreeData(item.children, targetPath, newItems),
      };
    }

    return item;
  });
};

const removeNodesFromTree = (tree: TreeViewItem[], nodeIds: string[]): TreeViewItem[] => {
  return tree.reduce((acc, item) => {
    if (nodeIds.includes(item.id)) return acc;

    if (item.children) {
      const newChildren = removeNodesFromTree(item.children, nodeIds);
      acc.push({ ...item, children: newChildren });
    } else {
      acc.push(item);
    }
    return acc;
  }, [] as TreeViewItem[]);
};

const addNodesToTree = (tree: TreeViewItem[], parentId: string, newNodes: TreeViewItem[]): TreeViewItem[] => {
  return tree.map(item => {
    if (item.id === parentId && item.children) {
      // Check for duplicates before adding
      const existingIds = new Set(item.children.map(child => child.id));
      const nodesToAdd = newNodes.filter(node => !existingIds.has(node.id));
      return { ...item, children: [...item.children, ...nodesToAdd] };
    }
    if (item.children) {
      return { ...item, children: addNodesToTree(item.children, parentId, newNodes) };
    }
    return item;
  });
};

type DialogMode = "create" | "upload" | "transfer" | "rename" | "conflict" | "transfer-conflict" | null;

// Types for conflict handling
interface ConflictFile {
  filename: string;
  path: string;
  size: number;
  created_at: string | null;
  updated_at: string | null;
}

interface ConflictData {
  message: string;
  conflicts: ConflictFile[];
  valid_files: number;
  total_attempted: number;
  target_path: string;
}

interface TransferConflictData {
  conflicts: string[];
}

// Last action tracking for info banner
interface LastAction {
  type: string;
  description: string;
  timestamp: number;
}

export default function FileTreeView() {
  const { user } = useAuth();
  const {
    canRenameSource,
    canDeleteSource,
    canUpload,
    canCreateRootFolderSource,
    canCreateFolderSource,
    canDeleteDestination,
    canCreateRootFolderDestination,
    canCreateFolderDestination,
    canTransfer,
    isSuperAdmin
  } = useFilePermissions();

  // Combined permissions for UI sections
  const hasSourceNewActions = canCreateRootFolderSource || canCreateFolderSource || canUpload;
  const hasDestinationNewActions = canCreateRootFolderDestination || canCreateFolderDestination;

  const [sourcePath, setSourcePath] = useState<string>("");
  const [destinationPath, setDestinationPath] = useState<string>("");
  const [pageSize] = useState(20);
  const [treeData, setTreeData] = useState<TreeViewItem[]>([]);
  const [showRecap, setShowRecap] = useState(false);
  const [showDialog, setShowDialog] = useState(false);
  const [dialogMode, setDialogMode] = useState<DialogMode>(null);
  const [folderName, setFolderName] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [selectedParent, setSelectedParent] = useState<TreeViewItem | null>(
    null
  );
  const [transferOperation, setTransferOperation] = useState<
    "move" | "undo" | "delete" | null
  >(null);
  const [transferId, setTransferId] = useState<string | null>(null);
  const [destinationTreeData, setDestinationTreeData] = useState<
    TreeViewItem[]
  >([]);
  const [selectedDestination, setSelectedDestination] =
    useState<TreeViewItem | null>(null);
  const [createTarget, setCreateTarget] = useState<"source" | "destination">(
    "source"
  );
  const [sourceSearch, setSourceSearch] = useState("");
  const [destinationSearch, setDestinationSearch] = useState("");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewFileName, setPreviewFileName] = useState<string>("");
  const [renameItem, setRenameItem] = useState<TreeViewItem | null>(null);
  const [newName, setNewName] = useState("");
  const [showSourceRootDialog, setShowSourceRootDialog] = useState(false);
  const [newSourceRoot, setNewSourceRoot] = useState("");
  const [previewIsPending, setPreviewIsPending] = useState(false);

  // Conflict state
  const [conflictData, setConflictData] = useState<ConflictData | null>(null);
  const [transferConflictData, setTransferConflictData] = useState<TransferConflictData | null>(null);
  const [uploadFilesRef, setUploadFilesRef] = useState<File[]>([]); // Store files to re-upload/replace

  // Last action state (for info banner only, no undo functionality)
  const [lastAction, setLastAction] = useState<LastAction | null>(null);

  const debouncedSourceSearch = useDebounce(sourceSearch, 500);
  const debouncedDestinationSearch = useDebounce(destinationSearch, 500);

  // Memoize category codes extraction to prevent recalculation
  const userCategoryCodes = useMemo(() => {
    if (
      !user?.assigned_file_categories ||
      user.assigned_file_categories.length === 0
    ) {
      return [];
    }
    return user.assigned_file_categories.map((cat) => cat.code);
  }, [user?.assigned_file_categories]);

  // Memoize regex patterns for better performance
  const categoryRegexPatterns = useMemo(() => {
    return createCategoryRegexPatterns(userCategoryCodes);
  }, [userCategoryCodes]);

  const { data: pathConfig, isLoading: pathConfigLoading } = useGetPathConfig();

  // Determine if we should fetch parents (only for root paths or when searching)
  // If searching, we use default behavior (fetch parents = true)
  // If no sourcePath or sourcePath is root, fetch parents = true
  // If subfolder, fetch parents = false (incremental update)
  const fetchSourceParents = useMemo(() => {
    if (debouncedSourceSearch) return true;
    if (!sourcePath || !pathConfig?.source_path) return true;

    // Normalize paths for comparison
    const current = sourcePath.endsWith('/') ? sourcePath : sourcePath + '/';
    const root = pathConfig.source_path.endsWith('/') ? pathConfig.source_path : pathConfig.source_path + '/';

    return current === root;
  }, [sourcePath, pathConfig?.source_path, debouncedSourceSearch]);

  const fetchDestinationParents = useMemo(() => {
    if (debouncedDestinationSearch) return true;
    if (!destinationPath || !pathConfig?.base_path) return true;

    // Normalize paths for comparison
    const current = destinationPath.endsWith('/') ? destinationPath : destinationPath + '/';
    const root = pathConfig.base_path.endsWith('/') ? pathConfig.base_path : pathConfig.base_path + '/';

    return current === root;
  }, [destinationPath, pathConfig?.base_path, debouncedDestinationSearch]);

  const {
    data: sourceData,
    isLoading: sourceIsLoading,
    error: sourceError,
    refetch: refetchSource,
    fetchNextPage: fetchNextSourcePage,
    hasNextPage: hasNextSourcePage,
    isFetchingNextPage: isFetchingNextSourcePage,
    isFetching: sourceIsFetching,
  } = useGCSListPathInfinite(sourcePath, pageSize, debouncedSourceSearch, fetchSourceParents);

  const {
    data: destinationData,
    isLoading: destinationIsLoading,
    error: destinationError,
    refetch: refetchDestination,
    fetchNextPage: fetchNextDestinationPage,
    hasNextPage: hasNextDestinationPage,
    isFetchingNextPage: isFetchingNextDestinationPage,
    isFetching: destinationIsFetching,
  } = useGCSListDestinationPathInfinite(
    destinationPath,
    pageSize,
    debouncedDestinationSearch,
    fetchDestinationParents
  );

  // Transfer progress
  const { data: progressData } = useTransferProgress(transferId);

  const transferProgressPercentage = useMemo(() => {
    if (!progressData) return 0;
    return (progressData.completed_items / Math.max(1, progressData.total_items)) * 100;
  }, [progressData]);

  // Mutations
  const updatePathConfigMutation = useUpdatePathConfig();
  const createFolderMutation = useCreateFolder();
  const uploadFilesMutation = useUploadFilesWithConflict(); // Use the new mutation
  const resolveConflictMutation = useResolveUploadConflict(); // Use the new mutation
  const transferFileMutation = useTransferFile();
  const transferFolderMutation = useTransferFolder();
  const deleteMutation = useDeleteGCS();
  const deleteDestinationMutation = useDeleteDestinationFile();
  const undoTransferMutation = useUndoTransfer(); // Add new mutation
  const checkTransferConflictsMutation = useCheckTransferConflicts();
  const previewMutation = useFilePreview();
  const renameFolderMutation = useRenameFolder();
  const renameFileMutation = useRenameFile();

  // Helper to track last action (for info banner only)
  const trackLastAction = useCallback((type: string, description: string) => {
    setLastAction({
      type,
      description,
      timestamp: Date.now(),
    });
  }, []);

  // Set source path from config on mount
  useEffect(() => {
    if (pathConfig?.base_path && pathConfig?.source_path) {
      setSourcePath(pathConfig.source_path);
      setDestinationPath(pathConfig.base_path);
      setNewSourceRoot(pathConfig.source_path);
    }
  }, [pathConfig]);

  useEffect(() => {
    if (!sourceData) return;

    try {
      const sourceItems =
        sourceData?.pages?.flatMap((page: any) => page.items || []) || [];

      const filteredItems =
        user?.role !== "superadmin"
          ? filterTreeDataByCategories(sourceItems, categoryRegexPatterns)
          : sourceItems;

      if (fetchSourceParents) {
        // Initial load or root refresh - replace tree
        setTreeData(filteredItems);
      } else {
        // Incremental update - merge into existing tree
        setTreeData(prevTree => mergeTreeData(prevTree, sourcePath, filteredItems));
      }
    } catch (error) {
      console.error("Failed to convert tree data:", error);
      toast.error("Failed to load file structure");
    }
  }, [sourceData, categoryRegexPatterns, user?.role, fetchSourceParents, sourcePath]);

  // Optimized destination data effect with memoization
  useEffect(() => {
    if (!destinationData) return;

    try {
      const destinationItems =
        destinationData?.pages?.flatMap((page: any) => page.items || []) || [];

      const filteredItems =
        user?.role !== "superadmin"
          ? filterTreeDataByCategories(destinationItems, categoryRegexPatterns)
          : destinationItems;

      if (fetchDestinationParents) {
        setDestinationTreeData(filteredItems);
      } else {
        setDestinationTreeData(prevTree => mergeTreeData(prevTree, destinationPath, filteredItems));
      }
    } catch (error) {
      console.error("Failed to convert destination tree:", error);
      toast.error("Failed to load destination file structure");
    }
  }, [destinationData, categoryRegexPatterns, user?.role, fetchDestinationParents, destinationPath]);

  // Show transfer progress
  useEffect(() => {
    if (progressData) {
      if (progressData.status === "completed") {
        toast.success("Transfer completed successfully");
        setTransferId(null);
      } else if (progressData.status === "failed") {
        toast.error(`Transfer failed: ${progressData.error}`);
        setTransferId(null);
      }
    }
  }, [progressData]);

  useEffect(() => {
    const handleScroll = (e: Event) => {
      const target = e.target as HTMLElement;
      if (!target.classList.contains("overflow-y-auto")) return;

      const scrollPercentage =
        (target.scrollTop + target.clientHeight) / target.scrollHeight;

      if (scrollPercentage > 0.85) {
        const isSourceTree = target.closest('[title="Source"]');

        if (isSourceTree && hasNextSourcePage && !isFetchingNextSourcePage) {
          fetchNextSourcePage();
        } else if (
          !isSourceTree &&
          hasNextDestinationPage &&
          !isFetchingNextDestinationPage
        ) {
          fetchNextDestinationPage();
        }
      }
    };

    const treeElements = document.querySelectorAll(".overflow-y-auto");
    treeElements.forEach((el) => el.addEventListener("scroll", handleScroll));

    return () => {
      treeElements.forEach((el) =>
        el.removeEventListener("scroll", handleScroll)
      );
    };
  }, [
    hasNextSourcePage,
    isFetchingNextSourcePage,
    fetchNextSourcePage,
    hasNextDestinationPage,
    isFetchingNextDestinationPage,
    fetchNextDestinationPage,
  ]);

  const customIconMap = useMemo(
    () => ({
      region: <Globe className="h-4 w-4 text-[#165D5D]" />,
      store: <Folder className="h-4 w-4 text-[#165D5D]" />,
      department: <FolderOpen className="h-4 w-4 text-[#165D5D]" />,
      item: <File className="h-4 w-4 text-[#8FB3A3]" />,
      folder: <Folder className="h-4 w-4 text-[#165D5D]" />,
      file: <File className="h-4 w-4 text-[#8FB3A3]" />,
      "sub folder": <FolderOpen className="h-4 w-4 text-[#165D5D]" />,
    }),
    []
  );

  const destinationIconMap = useMemo(
    () => ({
      region: <Globe className="h-4 w-4 text-[#165D5D]" />,
      store: <Folder className="h-4 w-4 text-[#165D5D]" />,
      department: <FolderOpen className="h-4 w-4 text-[#165D5D]" />,
      item: <File className="h-4 w-4 text-[#8FB3A3]" />,
      folder: <Folder className="h-4 w-4 text-[#165D5D]" />,
      file: <File className="h-4 w-4 text-[#8FB3A3]" />,
      "sub folder": <FolderOpen className="h-4 w-4 text-[#165D5D]" />,
    }),
    []
  );

  // Get all checked items recursively
  const getCheckedItems = useCallback(
    (items: TreeViewItem[]): TreeViewItem[] => {
      const checkedItems: TreeViewItem[] = [];

      const traverse = (itemList: TreeViewItem[]) => {
        itemList.forEach((item) => {
          if (item.checked) {
            checkedItems.push(item);
          } else if (item.children) {
            traverse(item.children);
          }
        });
      };

      traverse(items);
      return checkedItems;
    },
    []
  );

  const checkedItems = useMemo(
    () => getCheckedItems(treeData),
    [treeData, getCheckedItems]
  );

  // Update check state for items and their children
  const handleCheckChange = useCallback(
    (items: TreeViewItem | TreeViewItem[], checked: boolean) => {
      const itemsArray = Array.isArray(items) ? items : [items];

      const updateCheckState = (treeItems: TreeViewItem[]): TreeViewItem[] => {
        return treeItems.map((currentItem) => {
          if (itemsArray.some((item) => item.id === currentItem.id)) {
            return {
              ...currentItem,
              checked,
              children: currentItem.children
                ? updateAllChildren(currentItem.children, checked)
                : undefined,
            };
          }
          if (currentItem.children) {
            return {
              ...currentItem,
              children: updateCheckState(currentItem.children),
            };
          }
          return currentItem;
        });
      };

      const updateAllChildren = (
        children: TreeViewItem[],
        checked: boolean
      ): TreeViewItem[] => {
        return children.map((child) => ({
          ...child,
          checked,
          children: child.children
            ? updateAllChildren(child.children, checked)
            : undefined,
        }));
      };

      setTreeData((prevData) => updateCheckState(prevData));
    },
    []
  );

  // Update check state for destination items (single selection)
  const handleDestinationCheckChange = useCallback(
    (items: TreeViewItem | TreeViewItem[], checked: boolean) => {
      const itemsArray = Array.isArray(items) ? items : [items];

      const updateCheckState = (treeItems: TreeViewItem[]): TreeViewItem[] => {
        return treeItems.map((currentItem) => {
          // If we're turning it ON, uncheck everything else
          if (checked) {
            if (itemsArray.some((item) => item.id === currentItem.id)) {
              return { ...currentItem, checked };
            }
            if (currentItem.children) {
              return { ...currentItem, checked: false, children: updateCheckState(currentItem.children) };
            }
            return { ...currentItem, checked: false };
          } else {
            // Turning OFF just unchecks the specific item
            if (itemsArray.some((item) => item.id === currentItem.id)) {
              return { ...currentItem, checked };
            }
            if (currentItem.children) {
              return { ...currentItem, children: updateCheckState(currentItem.children) };
            }
            return currentItem;
          }
        });
      };

      setDestinationTreeData((prevData) => updateCheckState(prevData));

      if (checked && itemsArray.length > 0) {
        setSelectedDestination(itemsArray[0]);
      } else if (!checked && selectedDestination?.id === itemsArray[0]?.id) {
        setSelectedDestination(null);
      }
    },
    [selectedDestination]
  );

  const handleSourceSelection = useCallback((selectedItems: TreeViewItem[]) => {
    if (!selectedItems || selectedItems.length === 0) return;

    const item = selectedItems[0];

    if (item.children !== undefined && item.children.length === 0) {
      setSourcePath(item.id);
    }
  }, []);

  const handleDestinationSelection = useCallback(
    (selectedItems: TreeViewItem[]) => {
      if (!selectedItems || selectedItems.length === 0) return;

      const item = selectedItems[0];

      if (item.children !== undefined) {
        setSelectedDestination(item);
        toast.success(`Selected destination: ${item.name}`);

        if (item.children.length === 0) {
          setDestinationPath(item.id);
        }
      }
    },
    []
  );

  // Update resetDialog to include rename state
  const resetDialog = useCallback(() => {
    setShowDialog(false);
    setDialogMode(null);
    setFolderName("");
    setSelectedFiles([]);
    setSelectedParent(null);
    setTransferOperation(null);
    setRenameItem(null);
    setNewName("");
    setConflictData(null);
    setTransferConflictData(null);
    setUploadFilesRef([]);
  }, []);

  // Handle create folder
  const handleCreateFolder = useCallback(async (e?: React.FormEvent | React.MouseEvent) => {
    if (e) e.preventDefault();
    if (createFolderMutation.isPending) return;

    if (!folderName.trim()) {
      toast.info("Folder name cannot be empty");
      return;
    }

    try {
      // base paths according to target (do not change API shape)
      const basePath =
        createTarget === "source"
          ? newSourceRoot
          : newSourceRoot.replace(/[^/]+\/?$/, "");

      const currentPathToSend = selectedParent ? selectedParent.id : basePath;

      const result = await createFolderMutation.mutateAsync({
        folderName: folderName.trim(),
        currentPath: currentPathToSend,
      });

      // Track last action
      if (result) {
        trackLastAction("create", `Created folder: ${folderName.trim()}`);
      }
    } catch (error) {
      console.error("Folder creation failed:", error);
    } finally {
      resetDialog();
    }
  }, [
    folderName,
    createTarget,
    newSourceRoot,
    selectedParent,
    createFolderMutation,
    resetDialog,
    trackLastAction,
  ]);

  // Handle upload files
  const handleUploadFiles = useCallback(async (e?: React.FormEvent | React.MouseEvent) => {
    if (e) e.preventDefault();
    if (uploadFilesMutation.isPending) return;

    if (selectedFiles.length === 0) {
      toast.info("Please select files to upload");
      return;
    }

    try {
      await uploadFilesMutation.mutateAsync({
        path: selectedParent ? selectedParent.id : sourcePath,
        files: selectedFiles,
      });
      // Success is handled by onSuccess in hook
      resetDialog();
    } catch (error: any) {
      console.error("FileUpload failed:", error);

      // Handle Conflict (409)
      if (error.response && error.response.status === 409) {
        const data = error.response.data as ConflictData;
        setConflictData(data);
        setUploadFilesRef(selectedFiles); // Save files to use for replacement
        setDialogMode("conflict");
        // Don't reset dialog yet, switch to conflict mode
      } else {
        resetDialog();
      }
    }
  }, [selectedFiles, selectedParent, uploadFilesMutation, resetDialog]);

  // Handle conflict resolution
  const handleResolveConflict = useCallback(async (filePath: string, action: "keep" | "replace") => {
    if (!conflictData) return;

    const conflict = conflictData.conflicts.find(c => c.path === filePath);
    if (!conflict) return;

    // If replace, find the file object
    let fileToUpload: File | undefined;
    if (action === "replace") {
      fileToUpload = uploadFilesRef.find(f => f.name === conflict.filename);
      if (!fileToUpload) {
        toast.error(`Could not find file ${conflict.filename} to replace`);
        return;
      }
    }

    try {
      await resolveConflictMutation.mutateAsync({
        filePath: conflict.path,
        action: action,
        targetPath: conflictData.target_path,
        file: fileToUpload
      });

      // Remove the resolved conflict from the list
      setConflictData(prev => {
        if (!prev) return null;
        const newConflicts = prev.conflicts.filter(c => c.path !== filePath);
        if (newConflicts.length === 0) {
          // All resolved
          resetDialog();
          return null;
        }
        return { ...prev, conflicts: newConflicts };
      });

    } catch (error) {
      console.error("Failed to resolve conflict:", error);
    }
  }, [conflictData, uploadFilesRef, resolveConflictMutation, resetDialog]);

  // Optimized handleDelete
  const handleDelete = useCallback(
    async (items: TreeViewItem[], isDestination: boolean = false) => {
      if (items.length === 0) {
        toast.info("No items selected for deletion");
        return;
      }

      // Optimistic Update
      const itemIds = items.map(i => i.id);
      if (isDestination) {
        setDestinationTreeData(prev => removeNodesFromTree(prev, itemIds));
      } else {
        setTreeData(prev => removeNodesFromTree(prev, itemIds));
      }

      // Normalize paths: folders end with '/', files don't
      const paths = items.map((item) =>
        item.children !== undefined
          ? item.id.endsWith("/")
            ? item.id
            : `${item.id}/`
          : item.id.replace(/\/$/, "")
      );

      try {
        if (isDestination) {
          // Delete items from destination one by one using the special hook
          for (const item of items) {
            await deleteDestinationMutation.mutateAsync(item.id);
          }
        } else {
          await deleteMutation.mutateAsync(paths);
        }

        // We rely on delayed invalidation in hooks, but manual refetch ensures eventual consistency
        // Promise.all([refetchSource(), refetchDestination()]); 
        // Removing explicit refetch here as hooks handle it with delay
      } catch (error) {
        console.error("Delete failed:", error);
        // Ideally revert optimistic update here, but for now we rely on refetch to fix state
      }

      // Track last action
      trackLastAction("delete", `Deleted ${items.length} item(s)`);
    },
    [
      deleteMutation,
      deleteDestinationMutation,
      refetchSource,
      refetchDestination,
      trackLastAction,
    ]
  );

  const executeTransfer = useCallback(async (items: TreeViewItem[], destinationId: string) => {
    // Optimistic Update
    const itemIds = items.map(i => i.id);
    setTreeData(prev => removeNodesFromTree(prev, itemIds));

    if (transferOperation === "move") {
      const destPath = destinationId.endsWith('/') ? destinationId : destinationId + '/';
      const newDestItems = items.map(item => ({
        ...item,
        id: destPath + item.name,
        path: destPath + item.name,
        checked: false // Reset checked state
      }));
      setDestinationTreeData(prev => addNodesToTree(prev, destinationId, newDestItems));
    }

    try {
      // Separate files and folders
      const files = items.filter((item) => item?.type?.toLowerCase() === "file");
      const folders = items.filter(
        (item) => item?.type?.toLowerCase() !== "file"
      );

      // Transfer files
      files.forEach(async (file) => {
        await transferFileMutation.mutateAsync(
          {
            source: { path: file.id },
            destination: {
              root: destinationId,
              path: "",
            },
            operation: transferOperation === "move" ? "move" : "copy", // Default to move if not specified, though logic should prevent
          },
          {
            onSuccess: () => {
              refetchSource();
              refetchDestination();
            },
          }
        );
      });

      // Transfer folders
      folders.forEach(async (folder) => {
        await transferFolderMutation.mutateAsync(
          {
            sourceFolderPath: folder.id,
            destination: {
              root: destinationId,
              path: "",
            },
            operation: transferOperation === "move" ? "move" : "copy",
          },
          {
            onSuccess: (data) => {
              refetchSource();
              refetchDestination();
              setTransferId(data.transferId);
            },
          }
        );
      });

      // Track last action
      trackLastAction(
        "move",
        `Moved ${items.length} item(s) to ${selectedDestination?.name || 'destination'}`
      );

      toast.success(
        `Transferring ${items.length} item(s) to ${selectedDestination?.name || 'destination'}`
      );
      handleCheckChange(items, false);
      setSelectedDestination(null);
    } catch (error) {
      console.error("Transfer failed:", error);
    } finally {
      resetDialog();
    }
  }, [
    transferFileMutation,
    transferFolderMutation,
    transferOperation,
    refetchSource,
    refetchDestination,
    trackLastAction,
    selectedDestination,
    handleCheckChange,
    resetDialog
  ]);

  // Handle transfer
  const handleTransfer = useCallback(async (e?: React.FormEvent | React.MouseEvent) => {
    if (e) e.preventDefault();
    if (transferFileMutation.isPending || transferFolderMutation.isPending) return;

    if (!selectedDestination) {
      toast.error("Please select a destination folder");
      return;
    }

    const items = checkedItems;
    if (items.length === 0) {
      toast.error("No items selected for transfer");
      return;
    }

    try {
      if (transferOperation === "delete") {
        await handleDelete(items, false); // Delete from source
        toast.success(`Deleted ${items.length} item(s)`);
        handleCheckChange(items, false);
        setSelectedDestination(null);
        return;
      }

      if (transferOperation === "undo") {
        if (!transferId) {
          toast.error("No transfer to undo");
          return;
        }
        await undoTransferMutation.mutateAsync(transferId);
        // Success handled by hook
        return;
      }

      // Check for conflicts before transferring
      // Only check files for now as folder conflicts are more complex (merge)
      // Extract source paths of files
      const filePaths = items
        .filter(item => item.type?.toLowerCase() === 'file')
        .map(item => item.id);

      if (filePaths.length > 0) {
        const result = await checkTransferConflictsMutation.mutateAsync({
          source_paths: filePaths,
          destination: { root: selectedDestination.id, path: "" }
        });

        if (result.conflicts && result.conflicts.length > 0) {
          setTransferConflictData({ conflicts: result.conflicts });
          setDialogMode("transfer-conflict");
          // Don't reset dialog here, we are showing conflict dialog
          // But we need to close the 'transfer' dialog mode?
          // Since we reuse the dialog, just changing mode works.
          // Wait, 'transfer' mode was showing operation selection.
          // If we change mode, the dialog content updates.
          return;
        }
      }

      // No conflicts or user confirmed override (if called from elsewhere, but here it's initial call)
      await executeTransfer(items, selectedDestination.id);

    } catch (error) {
      console.error("Transfer initiation failed:", error);
      resetDialog();
    }
  }, [
    selectedDestination,
    checkedItems,
    transferOperation,
    transferId,
    handleDelete,
    undoTransferMutation,
    checkTransferConflictsMutation,
    executeTransfer,
    handleCheckChange,
    setSelectedDestination,
    resetDialog
  ]);



  // Handle file selection
  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files) {
        setSelectedFiles(Array.from(e.target.files));
      }
    },
    []
  );

  const closePreview = useCallback(() => {
    if (previewUrl && previewUrl.startsWith("blob:")) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setPreviewFileName("");
  }, [previewUrl]);

  const handlePreview = useCallback(
    async (item: TreeViewItem) => {
      if (item?.type?.toLowerCase() !== "file") {
        toast.info("Preview is only available for files");
        return;
      }

      setPreviewIsPending(true);

      try {
        const result = await previewMutation.mutateAsync(item.id);
        const previewData = result;

        if (previewData && previewData.preview_url) {
          const remoteUrl = previewData.preview_url;

          setPreviewFileName(previewData.filename || item.name);

          try {
            const response = await axios.get(remoteUrl, {
              responseType: "blob",
            });

            const blob = response.data;
            const objectUrl = URL.createObjectURL(blob);

            setPreviewUrl(objectUrl);
          } catch (fetchError) {
            console.error("Failed to fetch PDF with axios:", fetchError);
            toast.error(
              "Failed to load file content. Opening in new tab instead."
            );
            window.open(remoteUrl, "_blank");
            closePreview();
          }
        } else {
          toast.error("Preview URL not available");
        }
      } catch (error) {
        console.error("Preview error:", error);
      } finally {
        setPreviewIsPending(false);
      }
    },
    [previewMutation, closePreview]
  );

  // Add rename handler
  const handleRename = useCallback(async (e?: React.FormEvent | React.MouseEvent) => {
    if (e) e.preventDefault();
    if (renameFolderMutation.isPending || renameFileMutation.isPending) return;

    if (!renameItem || !newName.trim()) {
      toast.info("Name cannot be empty");
      return;
    }

    try {
      const oldName = renameItem.name;

      if (renameItem.type?.toLowerCase() === "file") {
        await renameFileMutation.mutateAsync({
          filePath: renameItem.id,
          newFileName: newName.trim(),
        });
      } else {
        await renameFolderMutation.mutateAsync({
          folderPath: renameItem.id,
          newFolderName: newName.trim(),
        });
      }

      // Track last action
      trackLastAction("rename", `Renamed: ${oldName} → ${newName.trim()}`);
    } catch (error) {
      console.error("Rename failed:", error);
    } finally {
      resetDialog();
    }
  }, [
    renameItem,
    newName,
    renameFileMutation,
    renameFolderMutation,
    resetDialog,
    trackLastAction,
  ]);

  const menuItems: TreeViewMenuItem[] = useMemo(
    () => [
      {
        id: "preview",
        label: "Preview",
        icon: <Eye className="h-4 w-4" />,
        action: (items) => {
          if (items.length === 1 && items[0].type?.toLowerCase() === "file") {
            handlePreview(items[0]);
          } else {
            toast.info("Please select a single file to preview");
          }
        },
      },
      ...(canRenameSource ? [{
        id: "rename",
        label: "Rename",
        icon: <Edit className="h-4 w-4" />,
        action: (items: TreeViewItem[]) => {
          if (items.length === 1) {
            const item = items[0];
            setRenameItem(item);

            if (item?.type?.toLowerCase() === "file") {
              const nameWithoutExt = item.name;
              setNewName(nameWithoutExt);
            } else {
              setNewName(item.name);
            }

            setDialogMode("rename");
            setShowDialog(true);
          } else {
            toast.info("Please select a single item to rename");
          }
        },
      }] : []),
      ...(canCreateFolderSource ? [{
        id: "create_folder",
        label: "Create Subfolder",
        icon: <FolderPlus className="h-4 w-4" />,
        action: (items: TreeViewItem[]) => {
          if (items.length === 1 && items[0].children) {
            setSelectedParent(items[0]);
            setCreateTarget("source");
            setDialogMode("create");
            setShowDialog(true);
          }
        },
      }] : []),
      ...(canUpload ? [{
        id: "upload_files",
        label: "Upload Files",
        icon: <Upload className="h-4 w-4" />,
        action: (items: TreeViewItem[]) => {
          if (items.length === 1 && items[0].children) {
            setSelectedParent(items[0]);
            setDialogMode("upload");
            setShowDialog(true);
          }
        },
      }] : []),
      ...(canTransfer ? [{
        id: "add_to_shipment",
        label: "Add to Selection",
        icon: <Share2 className="h-4 w-4" />,
        action: (items: TreeViewItem[]) => handleCheckChange(items, true),
      }] : []),
      ...(canDeleteSource ? [{
        id: "delete",
        label: "Delete",
        icon: <Trash2 className="h-4 w-4 text-red-500" />,
        action: (items: TreeViewItem[]) => handleDelete(items),
      }] : []),
    ],
    [handleCheckChange, handleDelete, handlePreview, canRenameSource, canCreateFolderSource, canUpload, canTransfer, canDeleteSource]
  );

  const downloadFile = (url: string, filename: string) => {
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleChangeSourceRoot = useCallback(async () => {
    if (!newSourceRoot.trim()) {
      toast.info("Source root path cannot be empty");
      return;
    }

    const formattedPath = newSourceRoot.trim().endsWith("/")
      ? newSourceRoot.trim()
      : `${newSourceRoot.trim()}/`;

    const pathParts = formattedPath.split("/").filter(Boolean);

    if (pathParts.length < 2) {
      toast.error(
        "Source root must include both base path and folder (e.g., Georgia 14/Pending Files/)"
      );
      return;
    }

    if (pathParts.length > 2) {
      toast.error(
        "Source root cannot be more than 2 levels deep (e.g., Georgia 14/Pending Files/)"
      );
      return;
    }

    const base_path = pathParts[0] + "/";
    const source_folder = pathParts[1];

    try {
      await updatePathConfigMutation.mutateAsync({
        base_path,
        source_folder,
      });
      setSourcePath(formattedPath);
      setShowSourceRootDialog(false);
      toast.success("Source root updated successfully");
    } catch (error) {
      // Error is handled by the mutation's onError
    }
  }, [newSourceRoot, updatePathConfigMutation]);

  const formatTimestamp = (timestamp: number) => {
    const now = Date.now();
    const diff = now - timestamp;
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (seconds < 60) return "Just now";
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return new Date(timestamp).toLocaleString();
  };


  return (
    <div className="w-full">
      {/* ── Success Notification Banner ── */}
      {lastAction && (
        <div className="w-full mb-5 flex border-4 border-[var(--unnamed-color-d2e38e)] bg-white rounded-[4px] overflow-hidden min-h-[48px]">
          {/* Left side colored block */}
          <div className="flex items-center justify-center bg-[var(--unnamed-color-d2e38e)] px-[18px]">
            <div className="bg-white rounded-full w-6 h-6 flex items-center justify-center shadow-sm">
              <Check className="w-4 h-4 text-black" strokeWidth={3} />
            </div>
          </div>
          {/* Right side content */}
          <div className="flex items-center justify-between flex-grow px-5 py-2">
            <span
              className="text-[14px] text-[var(--unnamed-color-000000)]"
              style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
            >
              Confirmation: {lastAction.description}
            </span>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setLastAction(null)}
              className="h-8 w-8 p-0 rounded-full hover:bg-gray-100 transition-opacity"
              aria-label="Close notification"
            >
              <X className="w-5 h-5 text-[var(--unnamed-color-000000)]" strokeWidth={1} />
            </Button>
          </div>
        </div>
      )}

      {/* ── Transfer Progress Modal ── */}
      <Dialog open={!!progressData && (progressData.status === "in_progress" || progressData.status === "pending")}>
        <DialogContent className="max-w-[760px] p-0 border-none bg-transparent shadow-none [&>button]:hidden">
          <div className="p-16 border border-[var(--unnamed-color-e2eae4)] rounded-sm bg-[var(--unnamed-color-ffffff)] flex flex-col items-center justify-center shadow-2xl">
            <div className="h-16 w-16 rounded-full bg-[var(--unnamed-color-eef4f4)] flex flex-col items-center justify-center mb-6">
              <Loader2 className="h-8 w-8 animate-spin text-[color:var(--unnamed-color-165d5d)]" />
            </div>

            <DialogTitle className="[font-family:var(--unnamed-font-family-montserrat)] font-bold text-[length:var(--unnamed-font-size-20)] text-[color:var(--unnamed-color-000000)] mb-10">
              Transferring Files
            </DialogTitle>

            <div className="w-full max-w-xl mx-auto">
              <div className="flex items-center justify-between mb-2">
                <span className="[font-family:var(--unnamed-font-family-montserrat)] font-bold text-[length:var(--unnamed-font-size-11)] text-[color:var(--unnamed-color-333333)]">
                  Progress
                </span>
                <span className="[font-family:var(--unnamed-font-family-montserrat)] font-semibold text-[length:var(--unnamed-font-size-11)] text-[color:var(--unnamed-color-57b6b2)]">
                  {Math.round(transferProgressPercentage)}%
                </span>
              </div>
              <Progress
                value={transferProgressPercentage}
                className="h-2 [&>div]:bg-[color:var(--unnamed-color-57b6b2)] bg-[var(--unnamed-color-f8f8f8)]"
              />
              <p className="text-center [font-family:var(--unnamed-font-family-montserrat)] font-medium text-[length:var(--unnamed-font-size-11)] text-[color:var(--unnamed-color-707070)] mt-6">
                Please wait while we transfer your files...
              </p>
            </div>
          </div>
          {/* Hidden description for accessibility */}
          <DialogDescription className="sr-only">
            File transfer is in progress. Please wait.
          </DialogDescription>
        </DialogContent>
      </Dialog>

      {/* ── Transfer Selection Card ── */}
      <div className="mb-6 p-6 border border-[var(--unnamed-color-e2eae4)] rounded-xl bg-[#E5F4F3] shadow-[0_4px_20px_rgba(0,93,94,0.06)]">
        <p className="mb-4 [font-family:var(--unnamed-font-family-georgia)] font-bold text-[length:var(--unnamed-font-size-16)] leading-[var(--unnamed-line-spacing-24)] text-[color:var(--unnamed-color-000000)]">
          Transfer Selection
        </p>
        <div className="flex items-center gap-4">
          {/* Source Selection Box */}
          <div className="flex-1 min-h-[56px] px-5 py-4 border border-[var(--unnamed-color-e2eae4)] rounded-lg bg-[var(--unnamed-color-ffffff)]">
            <p className="mb-2.5 [font-family:var(--unnamed-font-family-montserrat)] font-normal text-[length:var(--unnamed-font-size-11)] leading-[var(--unnamed-line-spacing-15)] text-[color:var(--unnamed-color-707070)]">
              Select the files and folders to transfer
            </p>
            <div className="flex flex-wrap gap-1.5">
              {checkedItems.length > 0 ? (
                checkedItems.slice(0, 5).map((item) => (
                  <span
                    key={item.id}
                    className="inline-flex items-center gap-1.5 rounded px-2.5 py-1.5 [font-family:var(--unnamed-font-family-montserrat)] font-bold text-[length:var(--unnamed-font-size-11)] bg-[var(--unnamed-color-eef4f4)] text-[color:var(--unnamed-color-165d5d)] shadow-sm"
                  >
                    {item.children !== undefined ? (
                      <Folder className="h-3 w-3" />
                    ) : (
                      <File className="h-3 w-3" />
                    )}
                    {item.name}{item.children ? ` (${item.children.length} Items)` : ""}
                  </span>
                ))
              ) : (
                <span className="inline-flex items-center gap-1.5 rounded px-2.5 py-1 [font-family:var(--unnamed-font-family-montserrat)] font-medium text-[length:var(--unnamed-font-size-11)] bg-[var(--unnamed-color-eef4f4)] text-[color:var(--unnamed-color-165d5d)]">
                  <Folder className="h-3 w-3" />
                  -
                </span>
              )}
              {checkedItems.length > 5 && (
                <span className="inline-flex items-center px-1 py-1 [font-family:var(--unnamed-font-family-montserrat)] font-medium text-[length:var(--unnamed-font-size-11)] text-[color:var(--unnamed-color-707070)]">
                  +{checkedItems.length - 5} more
                </span>
              )}
            </div>
          </div>

          {/* Arrow → */}
          <div className="flex items-center justify-center w-10 shrink-0">
            <svg width="24" height="20" viewBox="0 0 20 16" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 1L19 8L12 15M19 8H1" stroke="#165D5D" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>

          {/* Destination Selection Box */}
          <div className="flex-1 min-h-[56px] px-5 py-4 border border-[var(--unnamed-color-e2eae4)] rounded-lg bg-[var(--unnamed-color-ffffff)]">
            <p className="mb-2.5 [font-family:var(--unnamed-font-family-montserrat)] font-normal text-[length:var(--unnamed-font-size-11)] leading-[var(--unnamed-line-spacing-15)] text-[color:var(--unnamed-color-707070)]">
              Click on a folder to select destination
            </p>
            <div className="flex flex-wrap gap-1.5">
              {selectedDestination ? (
                <span className="inline-flex items-center gap-1.5 rounded px-2.5 py-1.5 [font-family:var(--unnamed-font-family-montserrat)] font-bold text-[length:var(--unnamed-font-size-11)] bg-[var(--dest-badge-bg)] text-[color:var(--dest-badge-text)] shadow-sm">
                  <Folder className="h-3 w-3 shadow-none" />
                  {selectedDestination.name}
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 rounded px-2.5 py-1 [font-family:var(--unnamed-font-family-montserrat)] font-medium text-[length:var(--unnamed-font-size-11)] bg-[var(--dest-badge-bg)] text-[color:var(--dest-badge-text)]">
                  <Folder className="h-3 w-3" />
                  -
                </span>
              )}
            </div>
          </div>

          {/* TRANSFER Button */}
          {canTransfer && (
            <Button
              className="shrink-0 flex items-center gap-2 [font-family:var(--unnamed-font-family-montserrat)] font-bold text-[length:var(--unnamed-font-size-11)] tracking-[0.1em] uppercase bg-[#165D5D] text-[color:var(--unnamed-color-ffffff)] rounded-[4px] h-9 px-5 hover:bg-[#1E4E55]"
              onClick={() => {
                if (checkedItems.length === 0) {
                  toast.info("Please select items to transfer");
                  return;
                }
                if (!selectedDestination) {
                  toast.error("Please select a destination folder first");
                  return;
                }
                setTransferOperation("move");
                setDialogMode("transfer");
                setShowDialog(true);
              }}
              disabled={
                checkedItems.length === 0 ||
                !selectedDestination ||
                transferFileMutation.isPending ||
                transferFolderMutation.isPending
              }
            >
              {transferFileMutation.isPending || transferFolderMutation.isPending ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Share2 className="h-3.5 w-3.5" />
              )}
              TRANSFER
            </Button>
          )}
        </div>
      </div>

      {/* ── Two-Panel Layout ── */}
      <div className="w-full flex flex-col xl:flex-row items-start gap-8">

        {/* ── LEFT: Staging (Processing) ── */}
        <div className="w-full xl:flex-1 min-w-0">
          <TreeView
            data={treeData}
            className="h-[500px] overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-gray-200 scrollbar-track-transparent"
            title="Staging (Processing)"
            titleIcon={
              <div className="flex items-center gap-2">
                <Info className="h-4.5 w-4.5 text-[#165D5D]" />
                {isSuperAdmin && (
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => {
                      setNewSourceRoot(sourcePath);
                      setShowSourceRootDialog(true);
                    }}
                    className="h-6 w-6 rounded-full text-[color:var(--unnamed-color-707070)]"
                  >
                    <Edit className="h-3 w-3" />
                  </Button>
                )}
              </div>
            }
            iconMap={customIconMap}
            showCheckboxes={true}
            showExpandAll={true}
            onSelectionChange={handleSourceSelection}
            onCheckChange={handleCheckChange}
            menuItems={menuItems}
            isLoading={pathConfigLoading || (sourceIsLoading && !treeData.length) || (!sourcePath && !treeData.length)}
            isItemLoading={sourceIsFetching}
            collapseOthersOnExpand={true}
          />
        </div>

        {/* ── RIGHT: Destination (Final) ── */}
        <div className="w-full xl:flex-1 min-w-0">
          <TreeView
            data={destinationTreeData}
            className="h-[500px] overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-gray-200 scrollbar-track-transparent"
            title="Destination (Final)"
            titleIcon={<Info className="h-4.5 w-4.5 text-[#165D5D]" />}
            iconMap={destinationIconMap}
            showCheckboxes={checkedItems.length > 0}
            showExpandAll={false}
            onSelectionChange={handleDestinationSelection}
            onCheckChange={handleDestinationCheckChange}
            menuItems={[
              {
                id: "preview_dest",
                label: "Preview",
                icon: <Eye className="h-4 w-4" />,
                action: (items: TreeViewItem[]) => {
                  if (items.length === 1 && items[0].type?.toLowerCase() === "file") {
                    handlePreview(items[0]);
                  } else {
                    toast.info("Please select a single file to preview");
                  }
                },
              },
              ...(canCreateFolderDestination ? [{
                id: "create_folder_dest",
                label: "Create Subfolder",
                icon: <FolderPlus className="h-4 w-4" />,
                action: (items: TreeViewItem[]) => {
                  if (items.length === 1 && items[0].children) {
                    setSelectedParent(items[0]);
                    setCreateTarget("destination");
                    setDialogMode("create");
                    setShowDialog(true);
                  }
                },
              }] : []),
              ...(canDeleteDestination ? [{
                id: "delete_folder_dest",
                label: "Delete",
                icon: <Trash2 className="h-4 w-4 text-red-500" />,
                action: (items: TreeViewItem[]) => handleDelete(items, true),
              }] : []),
            ]}
            isLoading={pathConfigLoading || (destinationIsLoading && !destinationTreeData.length) || (!destinationPath && !destinationTreeData.length)}
            isItemLoading={destinationIsFetching || destinationIsLoading}
            collapseOthersOnExpand={true}
          />
        </div>
      </div>

      {/* ══════════════════════════════════════════════
          DIALOGS — all preserved from original
         ══════════════════════════════════════════════ */}

      {/* Send Items Dialog */}
      <Dialog open={showRecap} onOpenChange={setShowRecap} modal={false}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>
              Sending {checkedItems.length} Item{checkedItems.length !== 1 ? "s" : ""}
            </DialogTitle>
            <DialogDescription>Review the items before sending</DialogDescription>
          </DialogHeader>
          <ScrollArea className="max-h-[60vh] mt-4">
            <div className="space-y-2">
              {checkedItems.map((item) => (
                <div key={item.id} className="flex items-center gap-2 p-3 border rounded-lg hover:bg-accent transition-colors">
                  {customIconMap[item?.type?.toLowerCase() as keyof typeof customIconMap]}
                  <div className="flex-1">
                    <span className="font-medium">{item.name}</span>
                    {item.metadata && (
                      <div className="text-xs text-muted-foreground">{(item.metadata.size / 1024 / 1024).toFixed(2)} MB</div>
                    )}
                  </div>
                  <span className="text-sm text-muted-foreground capitalize">{item?.type?.toLowerCase()}</span>
                </div>
              ))}
            </div>
          </ScrollArea>
          <DialogFooter className="mt-4">
            <Button variant="outline" onClick={() => setShowRecap(false)}>Cancel</Button>
            <Button onClick={() => {
              toast.success(`Successfully sent ${checkedItems.length} item(s)`);
              setShowRecap(false);
              handleCheckChange(checkedItems, false);
            }}>Confirm Send</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Create / Upload / Transfer / Rename Dialog */}
      <Dialog open={showDialog} onOpenChange={(open) => !open && resetDialog()} modal={false}>
        <DialogContent className={dialogMode === "transfer" ? "max-w-[720px] sm:max-w-[720px] p-0 gap-0 overflow-hidden" : ""}>
          <DialogHeader className={dialogMode === "transfer" ? "px-8 pt-6 pb-0 mb-0" : ""}>
            <DialogTitle className={dialogMode === "transfer" ? "[font-family:var(--unnamed-font-family-montserrat)] font-medium text-[28px] leading-[32px] text-[#1a1a1a]" : ""}>
              {dialogMode === "create" && "Create New Folder"}
              {dialogMode === "upload" && "Upload Files"}
              {dialogMode === "transfer" && "Confirm Transfer"}
              {dialogMode === "rename" && "Rename Item"}
            </DialogTitle>
            <DialogDescription className={dialogMode === "transfer" ? "[font-family:var(--unnamed-font-family-montserrat)] font-normal text-[16px] leading-[24px] text-[#000000]" : ""}>
              {dialogMode === "create" && "Add a new folder to the file system"}
              {dialogMode === "upload" && "Upload PDF files to the selected location"}
              {dialogMode === "transfer" && "Transfer the following items to the destination area?"}
              {dialogMode === "rename" && `Rename ${renameItem?.type?.toLowerCase() === "file" ? "file" : "folder"}`}
              {dialogMode === "conflict" && "File Conflict Detected"}
            </DialogDescription>
          </DialogHeader>

          <div className={dialogMode === "transfer" ? "px-8 pb-6 pt-3" : "space-y-4 py-4"}>
            {dialogMode === "conflict" && conflictData && (
              <div className="space-y-4">
                <div className="p-4 bg-yellow-50 dark:bg-yellow-950/20 border border-yellow-200 dark:border-yellow-800 rounded-lg flex items-start gap-3">
                  <AlertCircle className="h-5 w-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
                  <div className="space-y-1">
                    <h4 className="font-medium text-yellow-900 dark:text-yellow-100">Files already exist</h4>
                    <p className="text-sm text-yellow-700 dark:text-yellow-300">The following files already exist in the destination folder. Choose an action for each file.</p>
                  </div>
                </div>
                <ScrollArea className="max-h-[300px] border rounded-md">
                  <div className="p-4 space-y-4">
                    {conflictData.conflicts.map((conflict) => (
                      <div key={conflict.path} className="flex flex-col justify-between gap-4 p-3 bg-gray-50 dark:bg-gray-900 rounded-lg border">
                        <div className="space-y-1">
                          <div className="font-medium flex items-center gap-2">
                            <File className="h-4 w-4 text-gray-500" />
                            <span className="truncate line-clamp-1 block max-w-[300px]">{conflict.filename}</span>
                          </div>
                          <div className="text-xs text-muted-foreground flex items-center gap-3">
                            <span>Size: {(conflict.size / 1024).toFixed(1)} KB</span>
                            <span>•</span>
                            <span>Modified: {conflict.updated_at ? new Date(conflict.updated_at).toLocaleDateString() : "Unknown"}</span>
                          </div>
                        </div>
                        <div className="flex items-center gap-2 justify-end">
                          <Button variant="outline" size="sm" onClick={() => handleResolveConflict(conflict.path, "keep")} disabled={resolveConflictMutation.isPending}>Keep Existing</Button>
                          <Button size="sm" variant="destructive" onClick={() => handleResolveConflict(conflict.path, "replace")} disabled={resolveConflictMutation.isPending}>Replace</Button>
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </div>
            )}

            {dialogMode === "transfer-conflict" && transferConflictData && (
              <div className="space-y-4">
                <div className="p-4 bg-yellow-50 dark:bg-yellow-950/20 border border-yellow-200 dark:border-yellow-800 rounded-lg flex items-start gap-3">
                  <AlertCircle className="h-5 w-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
                  <div className="space-y-1">
                    <h4 className="font-medium text-yellow-900 dark:text-yellow-100">File Conflict Detected</h4>
                    <p className="text-sm text-yellow-700 dark:text-yellow-300">The selected file(s) already exist in the destination path. Do you want to override them or cancel?</p>
                  </div>
                </div>
                <ScrollArea className="max-h-[300px] border rounded-md">
                  <div className="p-4 space-y-2">
                    {transferConflictData.conflicts.map((filename, index) => (
                      <div key={index} className="flex items-center gap-2 text-sm p-2 bg-gray-50 rounded border">
                        <File className="h-4 w-4 text-gray-500" />
                        <span className="font-medium">{filename}</span>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
                <DialogFooter className="gap-2 sm:gap-0">
                  <Button variant="outline" onClick={() => resetDialog()}>Cancel</Button>
                  <Button variant="destructive" onClick={() => { if (selectedDestination) { executeTransfer(checkedItems, selectedDestination.id); } }}>Override</Button>
                </DialogFooter>
              </div>
            )}

            {dialogMode === "create" && (
              <>
                <div className="space-y-2">
                  <Label htmlFor="folder-name">Folder Name *</Label>
                  <Input id="folder-name" placeholder="Enter folder name" value={folderName} onChange={(e) => setFolderName(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && folderName.trim()) { handleCreateFolder(e); } }} autoFocus />
                </div>
                {selectedParent && (
                  <div className="p-3 bg-muted rounded-lg">
                    <div className="text-sm text-muted-foreground">Creating in: <span className="font-medium text-foreground">{selectedParent.name}</span></div>
                  </div>
                )}
              </>
            )}

            {dialogMode === "upload" && (
              <>
                <div className="space-y-2">
                  <Label htmlFor="file-upload">Select PDF Files *</Label>
                  <Input id="file-upload" type="file" accept=".pdf,application/pdf" multiple onChange={handleFileSelect} />
                  {selectedFiles.length > 0 && (<div className="text-sm text-muted-foreground">{selectedFiles.length} file(s) selected</div>)}
                </div>
                {selectedParent && (
                  <div className="p-3 bg-muted rounded-lg">
                    <div className="text-sm text-muted-foreground">Uploading to: <span className="font-medium text-foreground">{selectedParent.name}</span></div>
                  </div>
                )}
              </>
            )}

            {dialogMode === "transfer" && (
              <div className="space-y-1">
                {/* From → To transfer visualization */}
                <div className="flex items-center gap-4 p-5 rounded-[6px] bg-[#EFF8F8]">
                  {/* From Box */}
                  <div className="flex-1 min-w-0 border border-[#9AD1DC] rounded-[6px] bg-[#ffffff]">
                    <div className="px-6 py-5">
                      <p className="[font-family:var(--unnamed-font-family-montserrat)] font-normal text-[14px] leading-[18px] text-[#707070] mb-1.5">From</p>
                      <div className="flex items-center gap-3">
                        <div className="flex items-center justify-center w-[32px] h-[32px] rounded-[4px] bg-[#eef6f6] shrink-0">
                          <Folder className="h-5 w-5 fill-[#6fb6c9] stroke-[#6fb6c9]" />
                        </div>
                        <span className="[font-family:var(--unnamed-font-family-montserrat)] font-bold text-[16px] leading-[20px] text-[#1a1a1a] truncate">
                          Staging - ({checkedItems.length})
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Arrow */}
                  <div className="flex items-center justify-center shrink-0 w-12">
                    <svg width="30" height="24" viewBox="0 0 20 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M12 1L19 8L12 15M19 8H1" stroke="#165d5d" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </div>

                  {/* To Box */}
                  <div className="flex-1 min-w-0 border border-[#9AD1DC] rounded-[6px] bg-[#ffffff]">
                    <div className="px-6 py-5">
                      <p className="[font-family:var(--unnamed-font-family-montserrat)] font-normal text-[14px] leading-[18px] text-[#707070] mb-1.5">To</p>
                      <div className="flex items-center gap-3">
                        <div className="flex items-center justify-center w-[32px] h-[32px] rounded-[4px] bg-[#f8f9ea] shrink-0">
                          <Folder className="h-5 w-5 fill-[#c4d75d] stroke-[#c4d75d]" />
                        </div>
                        <span className="[font-family:var(--unnamed-font-family-montserrat)] font-bold text-[16px] leading-[20px] text-[#1a1a1a] truncate">
                          {selectedDestination ? selectedDestination.name : 'Destination folder'}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Info note */}
                <p className="[font-family:var(--unnamed-font-family-montserrat)] font-normal text-[14px] leading-[20px] text-[#000000]">
                  These item(s) will be queued for nightly batch processing. You may undo this transfer before processing begins.
                </p>
              </div>
            )}

            {dialogMode === "rename" && renameItem && (
              <>
                <div className="space-y-2">
                  <Label htmlFor="new-name">New {renameItem.type?.toLowerCase() === "file" ? "File" : "Folder"} Name *</Label>
                  <Input id="new-name" placeholder={`Enter new ${renameItem.type?.toLowerCase() === "file" ? "file" : "folder"} name`} value={newName} onChange={(e) => setNewName(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && newName.trim()) { handleRename(e); } }} autoFocus />
                  {renameItem.type?.toLowerCase() === "file" && (<div className="text-xs text-muted-foreground">Note: File extension will be preserved</div>)}
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <div className="text-sm text-muted-foreground">Current name: <span className="font-medium text-foreground">{renameItem.name}</span></div>
                  <div className="text-xs text-muted-foreground mt-1">Path: {renameItem.id}</div>
                </div>
              </>
            )}
          </div>

          {/* Transfer mode: custom footer matching screenshot */}
          {dialogMode === "transfer" && (
            <div className="flex items-center justify-end gap-4 px-8 pb-8">
              <button
                type="button"
                onClick={() => { resetDialog(); handleCheckChange(checkedItems, false); }}
                disabled={transferFileMutation.isPending || transferFolderMutation.isPending}
                className="px-8 py-3 border-2 border-[#b91c1c] rounded-[4px] bg-white [font-family:var(--unnamed-font-family-montserrat)] font-medium text-[15px] leading-[20px] tracking-[0.05em] text-[#000000] hover:bg-[#fef2f2] transition-colors disabled:opacity-50"
              >
                CANCEL
              </button>
              <button
                type="button"
                onClick={(e) => handleTransfer(e)}
                disabled={
                  transferFileMutation.isPending || transferFolderMutation.isPending ||
                  !selectedDestination
                }
                className="px-8 py-3 border-2 border-[#8FB3A3] rounded-[4px] bg-white [font-family:var(--unnamed-font-family-montserrat)] font-medium text-[15px] leading-[20px] tracking-[0.05em] text-[#000000] hover:bg-[#f2f9f8] transition-colors disabled:opacity-50"
              >
                {transferFileMutation.isPending || transferFolderMutation.isPending ? "CONFIRMING..." : "CONFIRM"}
              </button>
            </div>
          )}

          {/* Non-transfer modes: original footer */}
          {dialogMode !== "conflict" && dialogMode !== "transfer-conflict" && dialogMode !== "transfer" && (
            <DialogFooter>
              <Button variant="outline" onClick={() => { resetDialog(); handleCheckChange(checkedItems, false); }}>Cancel</Button>
              <Button
                onClick={(e) => {
                  if (dialogMode === "create") handleCreateFolder(e);
                  else if (dialogMode === "upload") handleUploadFiles(e);
                  else if (dialogMode === "rename") handleRename(e);
                }}
                disabled={
                  createFolderMutation.isPending || uploadFilesMutation.isPending ||
                  renameFolderMutation.isPending || renameFileMutation.isPending ||
                  (dialogMode === "create" && !folderName.trim()) ||
                  (dialogMode === "upload" && selectedFiles.length === 0) ||
                  (dialogMode === "rename" && !newName.trim())
                }
              >
                {dialogMode === "create" && (createFolderMutation.isPending ? "Creating..." : "Create")}
                {dialogMode === "upload" && (uploadFilesMutation.isPending ? "Uploading..." : "Upload")}
                {dialogMode === "rename" && (renameFolderMutation.isPending || renameFileMutation.isPending ? "Renaming..." : "Rename")}
              </Button>
            </DialogFooter>
          )}
        </DialogContent>
      </Dialog>

      {/* File Preview Dialog */}
      <Dialog open={!!previewUrl || previewIsPending} onOpenChange={(open) => !open && closePreview()} modal={false}>
        <DialogContent className="!max-w-[80vw] max-h-[90vh] p-0">
          <DialogHeader className="px-6 py-4 border-b">
            <DialogTitle className="flex items-center gap-2"><File className="h-5 w-5" />{previewFileName}</DialogTitle>
            <DialogDescription>File Preview - PDF Document</DialogDescription>
          </DialogHeader>
          <div className="relative w-full h-[calc(90vh-12rem)]">
            {previewIsPending ? (
              <div className="flex items-center justify-center h-full"><Loader2 className="h-8 w-8 animate-spin text-muted-foreground" /><span className="ml-2">Loading preview...</span></div>
            ) : previewUrl ? (
              <iframe src={previewUrl} className="w-full h-full border-0" title={previewFileName} onError={() => { toast.error("Failed to load preview"); closePreview(); }} />
            ) : (
              <div className="flex items-center justify-center h-full text-muted-foreground"><p>Preview not available</p></div>
            )}
          </div>
          <DialogFooter className="px-6 py-4 border-t">
            <Button variant="outline" onClick={closePreview} disabled={previewIsPending}>Close</Button>
            {previewUrl && (<Button onClick={() => downloadFile(previewUrl!, previewFileName)} disabled={previewIsPending}><Download className="h-4 w-4 mr-2" />Download</Button>)}
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Change Source Root Dialog */}
      <Dialog open={showSourceRootDialog} onOpenChange={setShowSourceRootDialog} modal={false}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Change Source Root</DialogTitle>
            <DialogDescription>Update the root path for source files. Path should end with /</DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="source-root">Source Root Path *</Label>
              <Input id="source-root" placeholder="e.g., Georgia 14/Pending Files/" value={newSourceRoot} onChange={(e) => setNewSourceRoot(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && newSourceRoot.trim()) { handleChangeSourceRoot(); } }} autoFocus />
              <div className="text-xs text-muted-foreground">
                Current root: <span className="font-mono">{pathConfig?.source_path || sourcePath}</span>
                {pathConfigLoading && (<Loader2 className="h-3 w-3 animate-spin inline ml-2" />)}
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => { setShowSourceRootDialog(false); setNewSourceRoot(pathConfig?.source_path || sourcePath); }} disabled={updatePathConfigMutation.isPending}>Cancel</Button>
            <Button onClick={handleChangeSourceRoot} disabled={!newSourceRoot.trim() || updatePathConfigMutation.isPending}>
              {updatePathConfigMutation.isPending ? (<><Loader2 className="h-4 w-4 animate-spin mr-2" />Updating...</>) : "Change Root"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div >
  );
}
