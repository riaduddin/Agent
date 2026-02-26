import { useState } from 'react';
import axiosInstance from '@/lib/axiosInstance';
import { toast } from 'sonner';

interface PreviewFile {
  path: string;
  name: string;
}

interface PreviewApiResponse {
  preview_url: string;
  filename: string;
  expires_at: string;
}

export const useFilePreview = () => {
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [previewFile, setPreviewFile] = useState<PreviewFile | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const openPreview = async (filePath: string, fileName: string) => {
    setLoading(true);
    setError(null);
    setPreviewFile({ path: filePath, name: fileName });
    setIsPreviewOpen(true);

    try {
      // Call backend API to get preview URL
      const response = await axiosInstance.get<PreviewApiResponse>(
        `/gcs/preview/${encodeURIComponent(filePath)}`
      );
      
      // Use the preview URL as returned by backend (should be full URL now)
      setPreviewUrl(response.data.preview_url);
    } catch (err: any) {
      console.error('Failed to get preview URL:', err);
      const errorMessage = err.response?.data?.message || 'Failed to load file preview';
      setError(errorMessage);
      toast.error('Preview Error', {
        description: errorMessage
      });
    } finally {
      setLoading(false);
    }
  };

  const closePreview = () => {
    setIsPreviewOpen(false);
    setPreviewFile(null);
    setPreviewUrl(null);
    setLoading(false);
    setError(null);
  };

  return {
    isPreviewOpen,
    previewFile,
    previewUrl,
    loading,
    error,
    openPreview,
    closePreview
  };
};
