'use client';

import React from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { AxiosError } from 'axios';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Download, RefreshCw } from 'lucide-react';
import axiosInstance from '@/lib/axiosInstance';

interface ChunkActionsProps {
    chunk: {
        id: string;
        status: string;
        start_page: number;
        end_page: number;
    };
    docId: string;
}

const getStatusDisplay = (status: string | undefined | null): { text: string; color: string } => {
    switch (status?.toLowerCase()) {
        case 'completed': return { text: 'Completed', color: 'text-green-600' };
        case 'processing': return { text: 'Processing', color: 'text-yellow-600' };
        case 'pending': return { text: 'Pending', color: 'text-blue-600' };
        case 'pending_classification':
        case 'error':
        case 'failed':
        case 'upload_failed':
        case 'ocr_failed':
        case 'embedding_failed':
        case 'vectorization_failed':
        case 'processing_error':
        case 'failed_embedding_dlq':
        case 'error_processing_chunk':
            return { text: status || 'Failed', color: 'text-red-600' };
        default: return { text: status || 'Unknown', color: 'text-gray-500' };
    }
};

export function ChunkActions({ chunk, docId }: ChunkActionsProps) {
    const queryClient = useQueryClient();

    const reprocessChunkMutation = useMutation<{ data: { msg: string } }, AxiosError, string>({
        mutationFn: (chunkId: string) => {
            return axiosInstance.post(`/docs/chunks/${chunkId}/reprocess`);
        },
        onSuccess: (data, chunkId) => {
            toast.success(`Successfully initiated reprocessing for chunk ${chunkId}.`);
            queryClient.invalidateQueries({ queryKey: ['documentChunks', docId] });
        },
        onError: (error: AxiosError, chunkId) => {
            let errorMsg = `Failed to reprocess chunk ${chunkId}.`;
            if (error.response?.data && typeof error.response.data === 'object' && 'msg' in error.response.data) {
                errorMsg = (error.response.data as { msg?: string }).msg || errorMsg;
            }
            toast.error(errorMsg);
        },
    });

    const handleDownloadChunk = async (chunkId: string) => {
        try {
            const response = await axiosInstance.get(`/docs/chunks/${chunkId}/download`);
            if (response.data.signed_url) {
                const link = document.createElement('a');
                link.href = response.data.signed_url;
                link.target = '_blank';
                link.setAttribute('download', response.data.filename || `chunk_${chunkId}.pdf`);
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                toast.success(`Downloading ${response.data.filename || `chunk_${chunkId}.pdf`}`);
            } else {
                toast.error("Failed to get download URL for chunk.");
            }
        } catch (error: unknown) {
            console.error("Error downloading chunk:", error);
            if (error instanceof AxiosError && error.response?.data?.msg) {
                toast.error(`Failed to download chunk: ${(error.response.data as { msg: string }).msg}`);
            } else {
                toast.error("Failed to download chunk due to an unexpected error.");
            }
        }
    };

    const handleReprocessChunk = (chunkId: string) => {
        reprocessChunkMutation.mutate(chunkId);
    };

    const chunkStatus = getStatusDisplay(chunk.status);

    return (
        <div className="flex gap-2 justify-end">
            {chunkStatus.color === 'text-red-600' && (
                <Button
                    variant="outline"
                    size="icon"
                    onClick={() => handleReprocessChunk(chunk.id)}
                    disabled={reprocessChunkMutation.isPending}
                    title="Reprocess Chunk"
                >
                    <RefreshCw className="h-4 w-4" />
                </Button>
            )}
            <Button
                variant="outline"
                size="icon"
                onClick={() => handleDownloadChunk(chunk.id)}
                title="Download Chunk"
            >
                <Download className="h-4 w-4" />
            </Button>
        </div>
    );
}
