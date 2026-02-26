import React from 'react';
import { Skeleton } from "@/components/ui/skeleton";

const FileBrowserSkeleton = () => {
  return (
    <div className="grid grid-cols-3 md:grid-cols-4 gap-4">
      {Array.from({ length: 12 }).map((_, index) => (
        <div key={index} className="text-center p-2">
          <Skeleton className="w-12 h-12 mx-auto" />
          <Skeleton className="h-4 w-20 mx-auto mt-1" />
        </div>
      ))}
    </div>
  );
};

export default FileBrowserSkeleton;
