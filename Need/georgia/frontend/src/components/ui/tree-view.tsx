/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/no-explicit-any */

"use client";

import { useState, useRef, useEffect, useCallback, useMemo, JSX } from "react";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from "@/components/ui/collapsible";
import {
  Folder,
  ChevronRight,
  Box,
  MoreVertical,
  Loader2,
  Check,
  Info,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";

export interface TreeViewItem {
  id: string;
  name: string;
  type: string;
  children?: TreeViewItem[];
  checked?: boolean;
  metadata?: any;
  created_date?: string;
  path?: string;
}

export interface TreeViewIconMap {
  [key: string]: React.ReactNode | undefined;
}

export interface TreeViewMenuItem {
  id: string;
  label: string;
  icon?: React.ReactNode;
  action: (items: TreeViewItem[]) => void;
}

export interface TreeViewProps {
  className?: string;
  data: TreeViewItem[];
  title?: string;
  showExpandAll?: boolean;
  showCheckboxes?: boolean;
  checkboxPosition?: "left" | "right";
  searchPlaceholder?: string;
  selectionText?: string;
  checkboxLabels?: {
    check: string;
    uncheck: string;
  };
  headerLabels?: {
    name?: string;
    date?: string;
    actions?: string;
  };
  getIcon?: (item: TreeViewItem, depth: number) => React.ReactNode;
  onSelectionChange?: (selectedItems: TreeViewItem[]) => void;
  onAction?: (action: string, items: TreeViewItem[]) => void;
  onCheckChange?: (item: TreeViewItem, checked: boolean) => void;
  iconMap?: TreeViewIconMap;
  menuItems?: TreeViewMenuItem[];
  showActionButton?: boolean;
  isLoading?: boolean;
  isItemLoading?: boolean;
  collapseOthersOnExpand?: boolean;
  containerClassName?: string;
  titleIcon?: React.ReactNode;
}

interface TreeItemProps {
  item: TreeViewItem;
  depth?: number;
  selectedIds: Set<string>;
  lastSelectedId: React.MutableRefObject<string | null>;
  onSelect: (ids: Set<string>) => void;
  expandedIds: Set<string>;
  onToggleExpand: (
    id: string,
    isOpen: boolean,
    collapseOthers?: boolean
  ) => void; // UPDATE THIS LINE
  getIcon?: (item: TreeViewItem, depth: number) => React.ReactNode;
  onAction?: (action: string, items: TreeViewItem[]) => void;
  onAccessChange?: (item: TreeViewItem, hasAccess: boolean) => void;
  allItems: TreeViewItem[];
  showAccessRights?: boolean;
  itemMap: Map<string, TreeViewItem>;
  iconMap?: TreeViewIconMap;
  menuItems?: TreeViewMenuItem[];
  getSelectedItems: () => TreeViewItem[];
  showActionButton?: boolean;
  isItemLoading?: boolean;
  collapseOthersOnExpand?: boolean; // ADD THIS LINE
}

const buildItemMap = (items: TreeViewItem[]): Map<string, TreeViewItem> => {
  const map = new Map<string, TreeViewItem>();
  const processItem = (item: TreeViewItem) => {
    map.set(item.id, item);
    item.children?.forEach(processItem);
  };
  items.forEach(processItem);
  return map;
};

const getCheckState = (
  item: TreeViewItem,
  itemMap: Map<string, TreeViewItem>
): "checked" | "unchecked" | "indeterminate" => {
  const originalItem = itemMap.get(item.id);
  if (!originalItem) return "unchecked";

  // For leaf nodes (no children), return their own checked state
  if (!originalItem.children || originalItem.children.length === 0) {
    return originalItem.checked ? "checked" : "unchecked";
  }

  // For parent nodes, only show checked if the parent itself is explicitly checked
  if (originalItem.checked) {
    return "checked";
  }

  // Otherwise, check if we should show indeterminate state
  let checkedCount = 0;
  let indeterminateCount = 0;
  originalItem.children.forEach((child) => {
    const childState = getCheckState(child, itemMap);
    if (childState === "checked") checkedCount++;
    if (childState === "indeterminate") indeterminateCount++;
  });

  // Show indeterminate if any children are checked or indeterminate
  if (checkedCount > 0 || indeterminateCount > 0) return "indeterminate";

  return "unchecked";
};

const defaultIconMap: TreeViewIconMap = {
  file: <Box className="h-4 w-4 text-red-600" />,
  folder: <Folder className="h-4 w-4 text-primary/80" />,
};

function TreeItem({
  item,
  depth = 0,
  selectedIds,
  lastSelectedId,
  onSelect,
  expandedIds,
  onToggleExpand,
  getIcon,
  onAction,
  onAccessChange,
  allItems,
  showAccessRights,
  itemMap,
  iconMap = defaultIconMap,
  menuItems,
  getSelectedItems,
  showActionButton = true,
  isItemLoading = false,
  collapseOthersOnExpand = false,
}: TreeItemProps): JSX.Element {
  const isOpen = expandedIds.has(item.id);
  const isSelected = selectedIds.has(item.id);
  const itemRef = useRef<HTMLDivElement>(null);

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    let newSelection = new Set(selectedIds);
    if (!itemRef.current) return;
    if (e.shiftKey && lastSelectedId.current !== null) {
      const items = Array.from(
        document.querySelectorAll("[data-tree-item]")
      ) as HTMLElement[];
      const lastIndex = items.findIndex(
        (el) => el.getAttribute("data-id") === lastSelectedId.current
      );
      const currentIndex = items.findIndex((el) => el === itemRef.current);
      const [start, end] = [
        Math.min(lastIndex, currentIndex),
        Math.max(lastIndex, currentIndex),
      ];
      items.slice(start, end + 1).forEach((el) => {
        const id = el.getAttribute("data-id");
        const parentFolderClosed = el.closest('[data-folder-closed="true"]');
        const isClosedFolder = el.getAttribute("data-folder-closed") === "true";
        if (id && (isClosedFolder || !parentFolderClosed)) {
          newSelection.add(id);
        }
      });
    } else if (e.ctrlKey || e.metaKey) {
      if (newSelection.has(item.id)) {
        newSelection.delete(item.id);
      } else {
        newSelection.add(item.id);
      }
    } else {
      newSelection = new Set([item.id]);
      if (item.children && isSelected) {
        onToggleExpand(item.id, !isOpen);
      }
    }
    lastSelectedId.current = item.id;
    onSelect(newSelection);
  };

  const handleAccessClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (onAccessChange) {
      const currentState = getCheckState(item, itemMap);
      const newChecked = currentState === "checked" ? false : true;
      onAccessChange(item, newChecked);
    }
  };

  const renderIcon = () => {
    if (getIcon) return getIcon(item, depth);
    const type = item?.type?.toLowerCase() || "folder";
    return (
      iconMap[type] ||
      iconMap.folder ||
      defaultIconMap.folder
    );
  };

  const renderActionButton = () => {
    if (!showActionButton || !menuItems || menuItems.length === 0) return null;

    return (
      <DropdownMenu modal={false}>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 opacity-100 group-hover:opacity-100 transition-opacity"
            onClick={(e) => e.stopPropagation()}
          >
            <MoreVertical className="h-4.5 w-4.5 text-[#165D5D]" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-48">
          {menuItems
            .filter((menuItem) => {
              if (menuItem.id === "preview") {
                return item.type?.toLowerCase() === "file";
              }
              if (
                menuItem.id === "create_folder" ||
                menuItem.id === "upload_files" ||
                menuItem.id === "create_folder_dest"
              ) {
                return item.children !== undefined;
              }
              return true;
            })
            .map((menuItem) => (
              <DropdownMenuItem
                key={menuItem.id}
                onClick={(e) => {
                  e.stopPropagation();
                  const items = selectedIds.has(item.id)
                    ? getSelectedItems()
                    : [item];
                  menuItem.action(items);
                }}
              >
                {menuItem.icon && (
                  <span className="mr-2 h-4 w-4">{menuItem.icon}</span>
                )}
                {menuItem.label}
              </DropdownMenuItem>
            ))}
        </DropdownMenuContent>
      </DropdownMenu>
    );
  };

  const renderMetadata = () => (
    <>
      <span className="font-sans font-normal text-[11px] leading-[15px] text-[#707070] text-end whitespace-nowrap">
        {item.created_date
          ? new Date(item.created_date)
            .toLocaleString("en-US", {
              month: "numeric",
              day: "numeric",
              year: "numeric",
              hour: "numeric",
              minute: "2-digit",
              hour12: true,
            })
          : "10/9/2025, 11:30 AM"}
      </span>
      <div className="flex justify-end pr-1">
        {renderActionButton()}
      </div>
    </>
  );

  return (
    <ContextMenu modal={false}>
      <ContextMenuTrigger>
        <div>
          <div
            ref={itemRef}
            data-tree-item
            data-id={item.id}
            data-depth={depth}
            data-folder-closed={item.children && !isOpen}
            className={cn(
              "select-none cursor-pointer px-1 border-b border-[#F8F8F8] hover:bg-[#F8F8F8]",
              item.checked || isSelected
                ? "bg-[#EEF8F7]"
                : "text-foreground"
            )}
            onClick={handleClick}
          >
            <div className="grid items-center h-10" style={{ gridTemplateColumns: `${showAccessRights ? '40px' : ''} minmax(0, 1fr) 180px 48px` }}>
              {showAccessRights && (
                <div className="flex items-center justify-center w-10 shrink-0" onClick={handleAccessClick}>
                  <div className={cn(
                    "w-5 h-5 rounded-[3px] border-2 transition-all flex items-center justify-center cursor-pointer",
                    getCheckState(item, itemMap) === "checked"
                      ? "bg-[#165D5D] border-[#165D5D]"
                      : (getCheckState(item, itemMap) === "indeterminate" ? "bg-white border-[#165D5D]" : "border-[#165D5D] bg-white")
                  )}>
                    {getCheckState(item, itemMap) === "checked" && <Check className="h-3.5 w-3.5 text-white" strokeWidth={4} />}
                    {getCheckState(item, itemMap) === "indeterminate" && <div className="w-2.5 h-0.5 bg-[#165D5D]" />}
                  </div>
                </div>
              )}

              <div className="flex items-center group min-w-0" style={{ paddingLeft: `${depth * 24}px` }}>
                {item.children ? (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 shrink-0 p-0 mr-1.5 hover:bg-transparent"
                    onClick={(e) => {
                      e.stopPropagation();
                      onToggleExpand(item.id, !isOpen, collapseOthersOnExpand);
                    }}
                  >
                    <motion.div initial={false} animate={{ rotate: isOpen ? 90 : 0 }} transition={{ duration: 0.1 }}>
                      <ChevronRight className="h-4.5 w-4.5 text-[#165D5D]" />
                    </motion.div>
                  </Button>
                ) : (
                  <div className="w-7.5 h-6 shrink-0" />
                )}
                <div className="shrink-0 mr-2.5">
                  {item.children ? (
                    <Folder className="h-5 w-5 text-[#8fb3a3] fill-[#8fb3a3]/20" />
                  ) : (
                    renderIcon()
                  )}
                </div>
                <span className="flex-1 font-sans font-normal text-[13px] leading-[15px] text-[#000000] truncate" title={item.name}>
                  {item.name}
                </span>
              </div>
              {renderMetadata()}
            </div>
          </div>

          {item.children && (
            <Collapsible
              open={isOpen}
              onOpenChange={(open) =>
                onToggleExpand(item.id, open, collapseOthersOnExpand)
              }
            >
              <AnimatePresence initial={false}>
                {isOpen && (
                  <CollapsibleContent forceMount asChild>
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.05 }}
                    >
                      {item.id === lastSelectedId.current &&
                        isItemLoading &&
                        !item.children.length
                        ? Array.from({
                          length: Math.floor(Math.random() * 7) + 3
                        }).map((_, idx) => (
                          <TreeItemSkeleton key={idx} level={depth + 1} showAccessRights={showAccessRights} />
                        ))
                        : item.children?.map((child) => (
                          <TreeItem
                            key={child.id}
                            item={child}
                            depth={depth + 1}
                            selectedIds={selectedIds}
                            lastSelectedId={lastSelectedId}
                            onSelect={onSelect}
                            expandedIds={expandedIds}
                            onToggleExpand={onToggleExpand}
                            getIcon={getIcon}
                            onAction={onAction}
                            onAccessChange={onAccessChange}
                            allItems={allItems}
                            showAccessRights={showAccessRights}
                            itemMap={itemMap}
                            iconMap={iconMap}
                            menuItems={menuItems}
                            getSelectedItems={getSelectedItems}
                            showActionButton={showActionButton}
                            isItemLoading={isItemLoading}
                            collapseOthersOnExpand={collapseOthersOnExpand}
                          />
                        ))}
                    </motion.div>
                  </CollapsibleContent>
                )}
              </AnimatePresence>
            </Collapsible>
          )}
        </div>
      </ContextMenuTrigger>
      <ContextMenuContent className="w-64">
        {menuItems
          ?.filter((menuItem) => {
            if (menuItem.id === "preview") {
              return item.type?.toLowerCase() === "file";
            }
            if (
              menuItem.id === "create_folder" ||
              menuItem.id === "upload_files" ||
              menuItem.id === "create_folder_dest"
            ) {
              return item.children !== undefined;
            }
            return true;
          })
          ?.map((menuItem) => (
            <ContextMenuItem
              key={menuItem.id}
              onClick={() => {
                const items = selectedIds.has(item.id)
                  ? getSelectedItems()
                  : [item];
                menuItem.action(items);
              }}
            >
              {menuItem.icon && (
                <span className="mr-2 h-4 w-4">{menuItem.icon}</span>
              )}
              {menuItem.label}
            </ContextMenuItem>
          ))}
      </ContextMenuContent>
    </ContextMenu>
  );
}


const TreeItemSkeleton = ({ level = 0, showAccessRights = true }: { level?: number, showAccessRights?: boolean }) => (
  <div
    className="grid items-center h-10 px-1 border-b border-[#F8F8F8] animate-pulse"
    style={{ gridTemplateColumns: `${showAccessRights ? '40px' : ''} minmax(0, 1fr) 180px 48px` }}
  >
    {showAccessRights && (
      <div className="flex items-center justify-center w-10">
        <div className="w-5 h-5 bg-gray-100 rounded-[3px] border-2 border-gray-200" />
      </div>
    )}
    <div className="flex items-center" style={{ paddingLeft: `${level * 24}px` }}>
      <div className="w-7.5 h-6 mr-1.5" /> {/* Chevron placeholder */}
      <div className="w-5 h-5 bg-gray-100 rounded mr-2.5" /> {/* Icon placeholder */}
      <div className="h-3 bg-gray-100 rounded w-32" /> {/* Name placeholder */}
    </div>
    <div className="flex justify-end">
      <div className="h-3 bg-gray-100 rounded w-28" /> {/* Date placeholder */}
    </div>
    <div className="flex justify-end pr-1">
      <div className="w-6 h-6 bg-gray-100 rounded" /> {/* Actions placeholder */}
    </div>
  </div>
);

const EmptyState = ({ searchQuery }: { searchQuery?: string }) => (
  <div className="flex flex-col items-center justify-center py-12 text-gray-500">
    <div className="text-4xl mb-3">📁</div>
    <p className="text-sm font-medium">
      {searchQuery ? "No items found" : "No items to display"}
    </p>
    {searchQuery && <p className="text-xs mt-1">Try adjusting your search</p>}
  </div>
);

export default function TreeView({
  className,
  checkboxLabels = { check: "Check", uncheck: "Uncheck" },
  data,
  iconMap,
  searchPlaceholder = "Search...",
  selectionText = "selected",
  showExpandAll = true,
  showCheckboxes = false,
  getIcon,
  onSelectionChange,
  onAction,
  onCheckChange,
  menuItems,
  showActionButton = true,
  isLoading = false,
  isItemLoading = false,
  collapseOthersOnExpand = false,
  headerLabels,
  title,
  containerClassName,
  titleIcon,
}: TreeViewProps) {
  const [currentMousePos, setCurrentMousePos] = useState<number>(0);
  const [dragStart, setDragStart] = useState<number | null>(null);
  const [dragStartPosition, setDragStartPosition] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [isDragging, setIsDragging] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState("");

  const dragRef = useRef<HTMLDivElement>(null);
  const lastSelectedId = useRef<string | null>(null);
  const treeRef = useRef<HTMLDivElement>(null);

  const DRAG_THRESHOLD = 10;
  const itemMap = useMemo(() => buildItemMap(data), [data]);

  const { filteredData, searchExpandedIds } = useMemo(() => {
    if (!searchQuery.trim()) {
      return { filteredData: data, searchExpandedIds: new Set<string>() };
    }
    const searchLower = searchQuery.toLowerCase();
    const newExpandedIds = new Set<string>();
    const itemMatches = (item: TreeViewItem): boolean => {
      const nameMatches = item.name.toLowerCase().includes(searchLower);
      if (nameMatches) return true;
      if (item.children)
        return item.children.some((child) => itemMatches(child));
      return false;
    };
    const filterTree = (items: TreeViewItem[]): TreeViewItem[] => {
      return items
        .map((item) => {
          if (!item.children) return itemMatches(item) ? item : null;
          const filteredChildren = filterTree(item.children);
          if (filteredChildren.length > 0 || itemMatches(item)) {
            if (item.children) newExpandedIds.add(item.id);
            return { ...item, children: filteredChildren };
          }
          return null;
        })
        .filter((item): item is TreeViewItem => item !== null);
    };
    return {
      filteredData: filterTree(data),
      searchExpandedIds: newExpandedIds,
    };
  }, [data, searchQuery]);

  useEffect(() => {
    if (searchQuery.trim()) {
      setExpandedIds((prev) => new Set([...prev, ...searchExpandedIds]));
    }
  }, [searchExpandedIds, searchQuery]);

  useEffect(() => {
    const handleClickAway = (e: MouseEvent) => {
      const target = e.target as Element;
      const clickedInside =
        (treeRef.current && treeRef.current.contains(target)) ||
        (dragRef.current && dragRef.current.contains(target)) ||
        target.closest('[role="menu"]') ||
        target.closest("[data-radix-popper-content-wrapper]");
      if (!clickedInside) {
        setSelectedIds(new Set());
        lastSelectedId.current = null;
      }
    };
    document.addEventListener("mousedown", handleClickAway);
    return () => document.removeEventListener("mousedown", handleClickAway);
  }, []);

  const getAllFolderIds = (items: TreeViewItem[]): string[] => {
    let ids: string[] = [];
    items.forEach((item) => {
      if (item.children) {
        ids.push(item.id);
        ids = [...ids, ...getAllFolderIds(item.children)];
      }
    });
    return ids;
  };

  const handleExpandAll = () => setExpandedIds(new Set(getAllFolderIds(data)));
  const handleCollapseAll = () => setExpandedIds(new Set());

  const getParentIds = (
    itemId: string,
    items: TreeViewItem[],
    parents: string[] = []
  ): string[] => {
    for (const item of items) {
      if (item.id === itemId) {
        return parents;
      }
      if (item.children) {
        const found = getParentIds(itemId, item.children, [
          ...parents,
          item.id,
        ]);
        if (
          found.length > 0 ||
          item.children.some((child) => child.id === itemId)
        ) {
          return item.children.some((child) => child.id === itemId)
            ? [...parents, item.id]
            : found;
        }
      }
    }
    return [];
  };

  const handleToggleExpand = (
    id: string,
    isOpen: boolean,
    collapseOthers = false
  ) => {
    const newExpandedIds = new Set(expandedIds);
    if (isOpen) {
      if (collapseOthers) {
        // Keep only parent folders expanded, collapse siblings
        const parentIds = getParentIds(id, data);
        const newSet = new Set(parentIds);
        newSet.add(id);
        setExpandedIds(newSet);
      } else {
        newExpandedIds.add(id);
        setExpandedIds(newExpandedIds);
      }
    } else {
      newExpandedIds.delete(id);
      setExpandedIds(newExpandedIds);
    }
  };

  const getSelectedItems = useCallback((): TreeViewItem[] => {
    const items: TreeViewItem[] = [];
    const processItem = (item: TreeViewItem) => {
      if (selectedIds.has(item.id)) items.push(item);
      item.children?.forEach(processItem);
    };
    data.forEach(processItem);
    return items;
  }, [selectedIds, data]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0 || (e.target as HTMLElement).closest("button")) return;
    setDragStartPosition({ x: e.clientX, y: e.clientY });
  }, []);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!(e.buttons & 1)) {
        setIsDragging(false);
        setDragStart(null);
        setDragStartPosition(null);
        return;
      }
      if (!dragStartPosition) return;
      const deltaX = e.clientX - dragStartPosition.x;
      const deltaY = e.clientY - dragStartPosition.y;
      const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
      if (!isDragging) {
        if (distance > DRAG_THRESHOLD) {
          setIsDragging(true);
          setDragStart(dragStartPosition.y);
          if (!e.shiftKey && !e.ctrlKey) {
            setSelectedIds(new Set());
            lastSelectedId.current = null;
          }
        }
        return;
      }
      if (!dragRef.current) return;
      const items = Array.from(
        dragRef.current.querySelectorAll("[data-tree-item]")
      ) as HTMLElement[];
      const startY = dragStart;
      const currentY = e.clientY;
      const [selectionStart, selectionEnd] = [
        Math.min(startY || 0, currentY),
        Math.max(startY || 0, currentY),
      ];
      const newSelection = new Set(
        e.shiftKey || e.ctrlKey ? Array.from(selectedIds) : []
      );
      items.forEach((item) => {
        const rect = item.getBoundingClientRect();
        const itemTop = rect.top;
        const itemBottom = rect.top + rect.height;
        if (itemBottom >= selectionStart && itemTop <= selectionEnd) {
          const id = item.getAttribute("data-id");
          const isClosedFolder =
            item.getAttribute("data-folder-closed") === "true";
          const parentFolderClosed = item.closest(
            '[data-folder-closed="true"]'
          );
          if (id && (isClosedFolder || !parentFolderClosed))
            newSelection.add(id);
        }
      });
      setSelectedIds(newSelection);
      setCurrentMousePos(e.clientY);
    },
    [isDragging, dragStart, selectedIds, dragStartPosition]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    setDragStart(null);
    setDragStartPosition(null);
  }, []);

  useEffect(() => {
    if (isDragging) {
      document.addEventListener("mouseup", handleMouseUp);
      document.addEventListener("mouseleave", handleMouseUp);
    }
    return () => {
      document.removeEventListener("mouseup", handleMouseUp);
      document.removeEventListener("mouseleave", handleMouseUp);
    };
  }, [isDragging, handleMouseUp]);

  useEffect(() => {
    if (onSelectionChange) onSelectionChange(getSelectedItems());
  }, [selectedIds, onSelectionChange, getSelectedItems]);

  return (
    <div className={cn("flex gap-4", containerClassName)} title={title}>
      <div
        ref={treeRef}
        className="bg-white rounded-[10px] space-y-4 w-full relative shadow-[0px_3px_6px_#005D5E2E]"
      >
        <div
          ref={dragRef}
          className={cn("rounded-lg bg-card relative select-none", className)}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
        >
          {isDragging && (
            <div
              className="absolute inset-0 bg-blue-500/0 pointer-events-none"
              style={{
                top: Math.min(
                  dragStart || 0,
                  dragStart === null ? 0 : currentMousePos
                ),
                height: Math.abs(
                  (dragStart || 0) - (dragStart === null ? 0 : currentMousePos)
                ),
              }}
            />
          )}

          {title && (
            <div className="flex items-center gap-2 mb-2 pl-4">
              <h2 className="text-left font-['Open_Sans'] font-bold text-[16px] leading-[35px] tracking-[0px] text-[#000000] opacity-100">
                {title}
              </h2>
              {titleIcon}
            </div>
          )}

          <div
            className="grid items-center h-[46px] px-1 border-y border-[#F8F8F8] bg-white"
            style={{
              gridTemplateColumns: `${showCheckboxes ? '40px' : ''} minmax(0, 1fr) 180px 48px`,
              background: '#FFFFFF 0% 0% no-repeat padding-box',
              opacity: 1
            }}
          >
            {showCheckboxes && <div className="w-10" />}
            <div className="flex items-center group min-w-0">
              <span className="flex-1 font-['Open_Sans'] font-medium text-[12px] leading-[46px] tracking-[0px] text-[#000000] text-left opacity-100">
                {headerLabels?.name ?? "Document Category"}
              </span>
            </div>
            <span className="font-['Open_Sans'] font-medium text-[12px] leading-[46px] tracking-[0px] text-[#000000] text-left whitespace-nowrap opacity-100">
              {headerLabels?.date ?? "Date Created"}
            </span>
            <span className="font-['Open_Sans'] font-medium text-[12px] leading-[46px] tracking-[0px] text-[#000000] text-left pr-1 opacity-100">
              {headerLabels?.actions ?? "Actions"}
            </span>
          </div>
          {isLoading ? (
            <>
              {Array.from({
                length: Math.floor(Math.random() * 8) + 5
              }).map((_, index) => {
                const randomLevel = Math.floor(Math.random() * 4); // 0, 1, 2, or 3
                return <TreeItemSkeleton key={index} level={randomLevel} showAccessRights={showCheckboxes} />;
              })}
            </>
          ) : filteredData.length === 0 ? (
            <EmptyState searchQuery={searchQuery} />
          ) : (
            filteredData.map((item) => (
              <TreeItem
                key={item.id}
                item={item}
                selectedIds={selectedIds}
                lastSelectedId={lastSelectedId}
                onSelect={setSelectedIds}
                expandedIds={expandedIds}
                onToggleExpand={handleToggleExpand}
                getIcon={getIcon}
                onAction={onAction}
                onAccessChange={onCheckChange}
                allItems={data}
                showAccessRights={showCheckboxes}
                itemMap={itemMap}
                iconMap={iconMap}
                menuItems={menuItems}
                getSelectedItems={getSelectedItems}
                showActionButton={showActionButton}
                isItemLoading={isItemLoading}
                collapseOthersOnExpand={collapseOthersOnExpand}
              />
            ))
          )}
        </div>
      </div>
    </div>
  );
}
