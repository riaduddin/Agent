import { useState } from 'react';
import axiosInstance from '@/lib/axiosInstance';
import { toast } from 'sonner';

interface CreateFolderRequest {
  folderName: string;
  currentPath: string;
}

interface CreateFolderResponse {
  message: string;
  folderPath: string;
}

export const useCreateFolder = (onSuccess?: () => void) => {
  const [isCreating, setIsCreating] = useState(false);

  const createFolder = async (request: CreateFolderRequest): Promise<CreateFolderResponse | null> => {
    setIsCreating(true);
    const toastId = toast.loading('Creating folder...');

    try {
      const response = await axiosInstance.post<CreateFolderResponse>('/gcs/create-folder', request);
      
      const result = response.data;
      
      toast.success(`Successfully created folder "${request.folderName}"!`, { id: toastId });

      if (onSuccess) {
        onSuccess();
      }

      return result;
    } catch (error: any) {
      console.error('Folder creation error:', error);
      const errorMessage = error.response?.data?.error || error.response?.data?.message || error.message;
      toast.error(`Error creating folder: ${errorMessage}`, { id: toastId });
      return null;
    } finally {
      setIsCreating(false);
    }
  };

  return {
    createFolder,
    isCreating
  };
};

export default useCreateFolder;
