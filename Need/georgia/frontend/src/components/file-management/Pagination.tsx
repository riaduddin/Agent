"use client";

import React from 'react';
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react';

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  pageSize: number;
  totalItems: number;
  totalFolders: number;
  totalFiles: number;
  hasNext: boolean;
  hasPrev: boolean;
  onPageChange: (page: number) => void;
  onPageSizeChange: (pageSize: number) => void;
  loading?: boolean;
}

const Pagination: React.FC<PaginationProps> = ({
  currentPage,
  totalPages,
  pageSize,
  totalItems,
  totalFolders,
  totalFiles,
  hasNext,
  hasPrev,
  onPageChange,
  onPageSizeChange,
  loading = false
}) => {
  const pageSizeOptions = [20, 50, 100];

  const getPageNumbers = () => {
    const pages = [];
    const maxVisiblePages = 5;
    
    if (totalPages <= maxVisiblePages) {
      // Show all pages if total is small
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      // Show pages around current page
      const start = Math.max(1, currentPage - 2);
      const end = Math.min(totalPages, currentPage + 2);
      
      if (start > 1) {
        pages.push(1);
        if (start > 2) pages.push('...');
      }
      
      for (let i = start; i <= end; i++) {
        pages.push(i);
      }
      
      if (end < totalPages) {
        if (end < totalPages - 1) pages.push('...');
        pages.push(totalPages);
      }
    }
    
    return pages;
  };

  if (totalItems === 0) {
    return null;
  }

  return (
    <div className="border-t border-gray-100 bg-gray-50 px-3 py-2">
      <div className="flex items-center justify-between text-xs">
        {/* Left: Items info */}
        <div className="text-gray-500 flex items-center gap-1">
          <span>{totalItems} items</span>
          <span className="text-gray-300">•</span>
          <span>{totalFolders}f {totalFiles}d</span>
        </div>

        {/* Center: Page navigation */}
        <div className="flex items-center gap-0.5">
          {/* First page */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onPageChange(1)}
            disabled={!hasPrev || loading}
            className="h-5 w-5 p-0 hover:bg-gray-200"
          >
            <ChevronsLeft className="h-2.5 w-2.5" />
          </Button>

          {/* Previous page */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onPageChange(currentPage - 1)}
            disabled={!hasPrev || loading}
            className="h-5 w-5 p-0 hover:bg-gray-200"
          >
            <ChevronLeft className="h-2.5 w-2.5" />
          </Button>

          {/* Page numbers - compact */}
          <div className="flex items-center gap-0.5 mx-1">
            {totalPages <= 5 ? (
              // Show all pages if 5 or fewer
              Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => (
                <Button
                  key={page}
                  variant="ghost"
                  size="sm"
                  onClick={() => onPageChange(page)}
                  disabled={loading}
                  className={`h-5 w-5 p-0 text-xs ${
                    page === currentPage 
                      ? "bg-blue-500 text-white hover:bg-blue-600" 
                      : "hover:bg-gray-200"
                  }`}
                >
                  {page}
                </Button>
              ))
            ) : (
              // Show current page with minimal context
              <>
                {currentPage > 2 && (
                  <>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => onPageChange(1)}
                      disabled={loading}
                      className="h-5 w-5 p-0 text-xs hover:bg-gray-200"
                    >
                      1
                    </Button>
                    {currentPage > 3 && <span className="text-gray-400 text-xs px-0.5">…</span>}
                  </>
                )}
                
                {/* Current page with neighbors */}
                {[currentPage - 1, currentPage, currentPage + 1]
                  .filter(page => page >= 1 && page <= totalPages)
                  .map((page) => (
                    <Button
                      key={page}
                      variant="ghost"
                      size="sm"
                      onClick={() => onPageChange(page)}
                      disabled={loading}
                      className={`h-5 w-5 p-0 text-xs ${
                        page === currentPage 
                          ? "bg-blue-500 text-white hover:bg-blue-600" 
                          : "hover:bg-gray-200"
                      }`}
                    >
                      {page}
                    </Button>
                  ))}

                {currentPage < totalPages - 1 && (
                  <>
                    {currentPage < totalPages - 2 && <span className="text-gray-400 text-xs px-0.5">…</span>}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => onPageChange(totalPages)}
                      disabled={loading}
                      className="h-5 w-5 p-0 text-xs hover:bg-gray-200"
                    >
                      {totalPages}
                    </Button>
                  </>
                )}
              </>
            )}
          </div>

          {/* Next page */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onPageChange(currentPage + 1)}
            disabled={!hasNext || loading}
            className="h-5 w-5 p-0 hover:bg-gray-200"
          >
            <ChevronRight className="h-2.5 w-2.5" />
          </Button>

          {/* Last page */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onPageChange(totalPages)}
            disabled={!hasNext || loading}
            className="h-5 w-5 p-0 hover:bg-gray-200"
          >
            <ChevronsRight className="h-2.5 w-2.5" />
          </Button>
        </div>

        {/* Right: Page size and info */}
        <div className="flex items-center gap-1 text-gray-500">
          <span className="text-xs">{currentPage}/{totalPages}</span>
          <Select
            value={pageSize.toString()}
            onValueChange={(value) => onPageSizeChange(parseInt(value))}
            disabled={loading}
          >
            <SelectTrigger className="w-8 h-4 text-xs border-0 bg-transparent p-0 hover:bg-gray-200 rounded">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {pageSizeOptions.map((size) => (
                <SelectItem key={size} value={size.toString()} className="text-xs">
                  {size}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
    </div>
  );
};

export default Pagination;
