/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-unused-vars */
"use client";

import React, { useState } from "react";
import withAuth from "@/components/auth/withAuth";
import { useQueryClient } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

import { useRouter } from "next/navigation";
import FileTreeView from "@/components/file-management/file-tree-view";

const FileManagementPage = () => {
  const router = useRouter();

  return (
    <div className="flex-1 bg-[var(--unnamed-color-ffffff)] p-8 overflow-y-auto">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <h1 className="[font-family:var(--unnamed-font-family-georgia)] font-bold text-[length:var(--unnamed-font-size-28)] leading-[var(--unnamed-line-spacing-32)] text-[color:var(--unnamed-color-000000)]">
              File Management
            </h1>
            <p className="mt-1 [font-family:var(--unnamed-font-family-montserrat)] font-normal text-[length:var(--unnamed-font-size-11)] leading-[var(--unnamed-line-spacing-15)] text-[color:var(--unnamed-color-707070)]">
              Organize, store and transfer your documents efficiently
            </p>
          </div>
          <Button
            variant="outline"
            className="[font-family:var(--unnamed-font-family-montserrat)] font-bold text-[11px] tracking-[0.1em] uppercase border-[var(--unnamed-color-165d5d)] text-[var(--unnamed-color-165d5d)] h-9 px-5 hover:bg-[var(--unnamed-color-eef4f4)] rounded-[4px]"
            onClick={() => router.push(`${process.env.NEXT_PUBLIC_BASE_PATH || '/r14'}/transfer-history`)}
          >
            VIEW TRANSFER HISTORY
          </Button>
        </div>

        {/* Content Section */}
        <div className="space-y-10">
          <FileTreeView />
        </div>
      </div>
    </div>
  );
};

export default withAuth(FileManagementPage);
