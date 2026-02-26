import { useState, useCallback } from 'react';

interface UsePaginationProps {
  initialPage?: number;
  initialPageSize?: number;
}

interface UsePaginationReturn {
  page: number;
  pageSize: number;
  setPage: (page: number) => void;
  setPageSize: (pageSize: number) => void;
  resetPagination: () => void;
  getQueryParams: () => { page: number; page_size: number };
}

const usePagination = ({
  initialPage = 1,
  initialPageSize = 50
}: UsePaginationProps = {}): UsePaginationReturn => {
  const [page, setPageState] = useState(initialPage);
  const [pageSize, setPageSizeState] = useState(initialPageSize);

  const setPage = useCallback((newPage: number) => {
    setPageState(Math.max(1, newPage));
  }, []);

  const setPageSize = useCallback((newPageSize: number) => {
    setPageSizeState(newPageSize);
    setPageState(1); // Reset to first page when changing page size
  }, []);

  const resetPagination = useCallback(() => {
    setPageState(initialPage);
    setPageSizeState(initialPageSize);
  }, [initialPage, initialPageSize]);

  const getQueryParams = useCallback(() => ({
    page,
    page_size: pageSize
  }), [page, pageSize]);

  return {
    page,
    pageSize,
    setPage,
    setPageSize,
    resetPagination,
    getQueryParams
  };
};

export default usePagination;
