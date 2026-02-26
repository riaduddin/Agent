// src/app/(app)/upload/page.tsx - Reverted to individual uploads for progress display
"use client";



import withAuth from "@/components/auth/withAuth";
import React, { useState, useCallback } from 'react';
import { useDropzone, FileRejection } from 'react-dropzone'; // Import FileRejection
import { isAxiosError } from 'axios'; // Import axios and isAxiosError
import axiosInstance from '@/lib/axiosInstance';
import { Progress } from "@/components/ui/progress"; // Import Progress component
import { Button } from "@/components/ui/button"; // Added Button import
import { UploadCloud, FileText, CheckCircle, AlertCircle, Loader2, X, AlertTriangle, Check } from 'lucide-react';
import { cn } from "@/lib/utils"; // Restored cn import
import { toast } from 'sonner';
// Removed useMutation as we upload individually now
// import { useMutation } from '@tanstack/react-query';

interface FileStatus {
  id: string;
  file: File;
  status: 'pending' | 'uploading' | 'success' | 'warning' | 'error';
  progress: number; // Use progress for individual uploads
  message?: string;
}

// No batch upload function needed

function UploadPage() {
  const [fileStatuses, setFileStatuses] = useState<FileStatus[]>([]);
  // Track uploading state locally
  const [isUploading, setIsUploading] = useState(false);
  const [showSuccessBanner, setShowSuccessBanner] = useState(false);

  // --- Move updateFileStatus inside the component ---
  const updateFileStatus = useCallback((id: string, updates: Partial<Omit<FileStatus, 'id' | 'file'>>) => {
    setFileStatuses(prevStatuses =>
      prevStatuses.map(fs => (fs.id === id ? { ...fs, ...updates } : fs))
    );
  }, []);

  // --- Re-implement uploadFile for single file upload ---
  const uploadFile = useCallback(async (fileStatus: FileStatus) => {
    const { id, file } = fileStatus;
    updateFileStatus(id, { status: 'uploading', progress: 0, message: 'Starting upload...' });
    setIsUploading(true); // Set uploading state
    setShowSuccessBanner(false); // Reset banner on new upload start

    const formData = new FormData();
    formData.append('file', file); // Send only one file

    try {
      const response = await axiosInstance.post('/docs/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = progressEvent.total
            ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
            : 0;
          // Update progress and message for the specific file
          updateFileStatus(id, { progress: percentCompleted, message: `Uploading... ${percentCompleted}%` });
        },
      });

      // Process response for the single file
      // Use Record<string, unknown> for response data items
      const errorData = response.data.failed_uploads?.find((err: Record<string, unknown>) => err.filename === file.name);
      const warningData = response.data.upload_warnings?.find((warn: Record<string, unknown>) => warn.filename === file.name);
      const successData = response.data.successful_uploads?.find((res: Record<string, unknown>) => res.filename === file.name);

      if (errorData) {
        updateFileStatus(id, { status: 'error', progress: 0, message: (errorData.error as string) || 'Upload or processing failed' }); // Added type assertion for errorData.error
      } else if (warningData) {
        updateFileStatus(id, { status: 'warning', progress: 100, message: warningData.warning || 'Upload succeeded with warnings' });
        setShowSuccessBanner(true); // Show banner on success/warning
      } else if (successData) {
        updateFileStatus(id, { status: 'success', progress: 100, message: 'Uploaded, pending processing.' });
        setShowSuccessBanner(true); // Show banner on success
      } else {
        if (response?.data?.successful_uploads?.length > 0) {
          updateFileStatus(id, { status: 'success', progress: 100, message: 'Uploaded successfully.' });
          setShowSuccessBanner(true); // Show banner on success
        } else {
          console.error("Unexpected backend response for file:", file.name, response.data);
          updateFileStatus(id, { status: 'error', progress: 0, message: 'Unknown status from server' });
        }
      }

    } catch (error: unknown) { // Use unknown type
      console.error("Upload error for file:", file.name, error);
      let errorMsg = 'Network or server error during upload';
      // Use type guard for AxiosError
      if (isAxiosError(error) && error.response?.data?.msg) {
        errorMsg = error.response.data.msg;
      } else if (error instanceof Error) {
        errorMsg = error.message;
      }
      updateFileStatus(id, {
        status: 'error',
        progress: 0,
        message: errorMsg,
      });
    } finally {
      // Check if any other files are still uploading after this one finishes/fails
      setFileStatuses(prev => {
        const stillUploading = prev.some(fs => fs.status === 'uploading' && fs.id !== id);
        setIsUploading(stillUploading);
        return prev;
      });
    }
  }, [updateFileStatus]); // Include updateFileStatus dependency


  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: FileRejection[]) => { // Replaced any[] with FileRejection[]
    const newFileStatuses: FileStatus[] = [];

    // Handle accepted PDF files
    acceptedFiles.forEach(file => {
      const fileId = `${file.name}-${Date.now()}`;
      newFileStatuses.push({
        id: fileId,
        file: file,
        status: 'pending',
        progress: 0,
      });
    });

    // Handle rejected files
    rejectedFiles.forEach(rejectedFile => {
      const fileId = `${rejectedFile.file.name}-${Date.now()}`;
      let message = 'File type not supported.';
      // Use a more specific type for rejectedFile errors if available
      if (rejectedFile.errors && Array.isArray(rejectedFile.errors) && rejectedFile.errors.length > 0) {
        message = rejectedFile.errors.map((e: { message: string }) => e.message).join(', ');
      }
      newFileStatuses.push({
        id: fileId,
        file: rejectedFile.file,
        status: 'error',
        progress: 0,
        message: message,
      });
    });

    // Add new files to the list
    setFileStatuses(prevStatuses => [...prevStatuses, ...newFileStatuses]);

    // Trigger individual uploads for pending files
    newFileStatuses.forEach(fs => {
      if (fs.status === 'pending') {
        uploadFile(fs); // Call uploadFile for each pending file
      }
    });

  }, [uploadFile]); // Include uploadFile dependency

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    noClick: true,
    noKeyboard: true,
    multiple: true,
    accept: {
      'application/pdf': ['.pdf'],
    },
  });

  // Function to remove a file from the list
  const removeFile = (id: string) => {
    // Prevent removing during upload
    setFileStatuses(prevStatuses => prevStatuses.filter(fs => {
      if (fs.id === id && fs.status === 'uploading') {
        toast.warning("Cannot remove file while it's uploading.");
        return true; // Keep the file
      }
      return fs.id !== id; // Remove the file
    }));
  };

  return (
    <div className="flex flex-col items-center  w-full min-h-[calc(100vh-64px)] bg-white p-6">
      {/* Success Banner */}
      {showSuccessBanner && (
        <div className="w-full mb-8 flex border-2 border-[#CBE086] bg-white overflow-hidden min-h-[48px] max-w-5xl">
          {/* Left side colored block */}
          <div className="flex items-center justify-center bg-[#CBE086] px-[18px]">
            <div className="bg-white rounded-full w-6 h-6 flex items-center justify-center shadow-sm">
              <Check className="w-4 h-4 text-black" strokeWidth={3} />
            </div>
          </div>
          {/* Right side content */}
          <div className="flex items-center justify-between flex-grow px-5 py-2">
            <span className="text-[14px] text-[#000000]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Confirmation: Document(s) uploaded successfully.</span>
            <button onClick={() => setShowSuccessBanner(false)} className="hover:opacity-70 transition-opacity">
              <X className="w-5 h-5 text-[#000000]" strokeWidth={1} />
            </button>
          </div>
        </div>
      )}

      <div className="w-full max-w-4xl flex flex-col items-center mt-20">
        {/* Hero Heading (Georgia) */}
        <h1
          className="text-[24px] font-bold text-[#255c5d] mb-10 text-center"
          style={{ fontFamily: 'var(--unnamed-font-family-georgia)' }}
        >
          Hi! Let’s start uploading your files.
        </h1>

        {/* Drag and Drop Area */}
        <div
          {...getRootProps()}
          className={cn(
            "w-full aspect-[3.25/1] border border-dashed rounded-[6px] flex flex-col items-center justify-center text-center relative transition-all duration-200 ease-in-out cursor-pointer",
            isDragActive
              ? "border-[#255C5D] bg-[#E2EAE4]"
              : "border-[#255C5D] bg-[#EEF4F4]"
          )}
        >
          <input {...getInputProps()} />

          <div className="flex flex-col items-center gap-3">
            <Button
              type="button"
              onClick={open}
              disabled={isUploading}
              className="bg-[#1E4E55] hover:bg-[#143B40] text-white px-10 h-[42px] rounded-[4px] uppercase text-[12px] font-bold tracking-wider w-[315px]"
              style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
            >
              Browse Files
            </Button>

            <p
              className="text-[14px] font-bold text-[#000000]"
              style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
            >
              Drag and drop PDF files, or select Browse Files
            </p>
          </div>
        </div>

        {/* Support Text */}
        <div className="w-full mt-2 flex justify-start">
          <p
            className="text-[12px] text-[#707070] font-medium"
            style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
          >
            Only PDF files are supported
          </p>
        </div>

        {/* Upload Status List (Queue) */}
        {/* {fileStatuses.length > 0 && (
          <div className="w-full mt-12 space-y-3">
            <h3
              className="text-[14px] font-bold text-[#255C5D] uppercase tracking-wider mb-4 px-2"
              style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
            >
              Upload Queue
            </h3>
            {fileStatuses.map((fs) => (
              <div
                key={fs.id}
                className="flex items-center p-4 bg-white border border-[#E0E0E0] rounded-[4px] space-x-4 shadow-sm hover:shadow-md transition-shadow"
              >
                <div className="p-2 bg-[#F1F8F8] rounded-[4px]">
                  <FileText className="w-5 h-5 text-[#255c5d]" />
                </div>

                <div className="flex-grow overflow-hidden">
                  <div className="flex items-center justify-between mb-1">
                    <p className="text-sm font-medium text-[#000000] truncate max-w-[70%]" title={fs.file.name}>
                      {fs.file.name}
                    </p>
                    <span className="text-[10px] text-[#707070]">
                      {(fs.file.size / 1024).toFixed(1)} KB
                    </span>
                  </div>

                  {fs.status === 'uploading' ? (
                    <div className="space-y-1">
                      <Progress value={fs.progress} className="h-1 bg-[#E0E0E0]" />
                      <p className="text-[10px] text-[#255c5d] font-medium">{fs.message || `Uploading... ${fs.progress}%`}</p>
                    </div>
                  ) : (
                    <p className={cn(
                      "text-[10px] font-medium",
                      fs.status === 'success' ? "text-[#255C5D]" :
                        fs.status === 'error' ? "text-red-600" :
                          fs.status === 'warning' ? "text-yellow-600" : "text-[#707070]"
                    )}>
                      {fs.message || fs.status.charAt(0).toUpperCase() + fs.status.slice(1)}
                    </p>
                  )}
                </div>

                <div className="flex items-center gap-3">
                  {fs.status === 'uploading' && <Loader2 className="w-4 h-4 text-[#255c5d] animate-spin" />}
                  {fs.status === 'success' && <CheckCircle className="w-4 h-4 text-[#255C5D]" />}
                  {fs.status === 'warning' && <AlertTriangle className="w-4 h-4 text-yellow-500" />}
                  {fs.status === 'error' && <AlertCircle className="w-4 h-4 text-red-600" />}

                  <button
                    onClick={() => removeFile(fs.id)}
                    disabled={fs.status === 'uploading'}
                    className="p-1 hover:bg-gray-100 rounded-full transition-colors disabled:opacity-30"
                  >
                    <X className="w-4 h-4 text-[#707070]" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )} */}
      </div>
    </div>
  );
}

// Restore withAuth HOC
export default withAuth(UploadPage);
