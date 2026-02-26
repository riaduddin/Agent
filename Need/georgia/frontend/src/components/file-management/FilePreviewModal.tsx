import React, { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { X, Loader2, AlertCircle, ExternalLink } from 'lucide-react';

interface FilePreviewModalProps {
  isOpen: boolean;
  filePath: string | null;
  fileName: string | null;
  previewUrl: string | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
}

const FilePreviewModal: React.FC<FilePreviewModalProps> = ({
  isOpen,
  filePath,
  fileName,
  previewUrl,
  loading,
  error,
  onClose
}) => {
  const [iframeError, setIframeError] = useState(false);
  const [permissionDenied, setPermissionDenied] = useState(false);
  const [checkingPermissions, setCheckingPermissions] = useState(false);
  // Handle ESC key to close modal and reset iframe error state
  useEffect(() => {
    const handleEscKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscKey);
      // Prevent body scrolling when modal is open
      document.body.style.overflow = 'hidden';
      // Reset iframe error state when modal opens
      setIframeError(false);
      setPermissionDenied(false);
      setCheckingPermissions(false);
    }

    return () => {
      document.removeEventListener('keydown', handleEscKey);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, onClose]);

  // Function to check iframe permissions
  const checkIframePermissions = async () => {
    try {
      // Check if the Permissions API is available
      if ('permissions' in navigator) {
        // Check for iframe/embed permissions (not all browsers support this)
        const result = await navigator.permissions.query({ name: 'camera' as PermissionName }); // Using camera as a proxy
        return result.state === 'granted';
      }
      return true; // Assume allowed if API not available
    } catch (error) {
      return true; // Assume allowed if error
    }
  };

  // Function to request iframe permissions
  const requestIframePermissions = async () => {
    setCheckingPermissions(true);
    try {
      // For iframe embedding, we need to check browser settings
      // Chrome doesn't have a direct API for iframe permissions, but we can:
      // 1. Try to create a test iframe
      // 2. Detect if it loads successfully
      // 3. Guide user to enable iframe permissions

      const testIframe = document.createElement('iframe');
      testIframe.style.display = 'none';
      testIframe.src = 'data:text/html,<html><body>Test</body></html>';

      return new Promise((resolve) => {
        testIframe.onload = () => {
          document.body.removeChild(testIframe);
          setPermissionDenied(false);
          setIframeError(false);
          resolve(true);
        };

        testIframe.onerror = () => {
          document.body.removeChild(testIframe);
          setPermissionDenied(true);
          resolve(false);
        };

        document.body.appendChild(testIframe);

        // Timeout after 3 seconds
        setTimeout(() => {
          if (document.body.contains(testIframe)) {
            document.body.removeChild(testIframe);
            setPermissionDenied(true);
            resolve(false);
          }
        }, 3000);
      });

    } catch (error) {
      console.error('Error checking iframe permissions:', error);
      setPermissionDenied(true);
      return false;
    } finally {
      setCheckingPermissions(false);
    }
  };

  if (!isOpen) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 bg-black bg-opacity-75 flex items-center justify-center">
      <div className="bg-white rounded-lg w-full h-full max-w-7xl max-h-[95vh] m-4 flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex justify-between items-center p-4 border-b bg-gray-50 rounded-t-lg">
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-semibold text-gray-900 truncate" title={fileName || ''}>
              {fileName || 'File Preview'}
            </h3>
            {filePath && (
              <p className="text-sm text-gray-500 truncate mt-1" title={filePath}>
                {filePath}
              </p>
            )}
          </div>
          <Button
            onClick={onClose}
            variant="ghost"
            size="sm"
            className="ml-4 hover:bg-gray-200"
            title="Close preview (ESC)"
          >
            <X className="h-5 w-5" />
          </Button>
        </div>

        {/* Content */}
        <div className="flex-1 relative bg-gray-100">
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-white">
              <div className="text-center">
                <Loader2 className="h-8 w-8 animate-spin text-blue-600 mx-auto mb-4" />
                <p className="text-gray-600">Loading preview...</p>
              </div>
            </div>
          )}

          {error && (
            <div className="absolute inset-0 flex items-center justify-center bg-white">
              <div className="text-center max-w-md mx-auto p-6">
                <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
                <h4 className="text-lg font-semibold text-gray-900 mb-2">Preview Error</h4>
                <p className="text-gray-600 mb-4">{error}</p>
                <div className="space-x-3">
                  <Button onClick={onClose} variant="outline">
                    Close
                  </Button>
                </div>
              </div>
            </div>
          )}

          {previewUrl && !loading && !error && (
            <>
              {!iframeError ? (
                <iframe
                  src={previewUrl}
                  className="w-full h-full border-0"
                  title={`Preview of ${fileName}`}
                  onLoad={() => {
                    setIframeError(false);
                    setPermissionDenied(false);
                  }}
                  onError={() => {
                    setIframeError(true);
                    checkIframePermissions().then(hasPermission => {
                      if (!hasPermission) {
                        setPermissionDenied(true);
                      }
                    });
                  }}
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <div className="text-center">
                    <div className="mb-4">
                      <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                        <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                      </div>
                      <h3 className="text-lg font-semibold text-gray-900 mb-2">
                        {permissionDenied ? 'Chrome Permissions Required' : 'Preview Not Available'}
                      </h3>
                      <p className="text-gray-600 mb-6">
                        {permissionDenied
                          ? 'Chrome is blocking iframe content. You can enable it in browser settings or open in a new tab.'
                          : 'Unable to preview in modal. Click below to open in a new tab instead.'
                        }
                      </p>
                    </div>
                    <div className="space-y-3">
                      {permissionDenied && (
                        <>
                          <Button
                            onClick={requestIframePermissions}
                            disabled={checkingPermissions}
                            className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 flex items-center gap-2 mx-auto"
                          >
                            {checkingPermissions ? (
                              <Loader2 className="w-4 h-4 animate-spin" />
                            ) : (
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                              </svg>
                            )}
                            {checkingPermissions ? 'Checking...' : 'Try Enable Preview'}
                          </Button>
                          <div className="text-xs text-gray-500 max-w-md mx-auto">
                            <p className="mb-2"><strong>To enable iframe previews in Chrome:</strong></p>
                            <ol className="list-decimal list-inside space-y-1 text-left">
                              <li>Click the lock/shield icon in the address bar</li>
                              <li>Allow "Insecure content" or "Mixed content"</li>
                              <li>Refresh the page and try preview again</li>
                            </ol>
                          </div>
                        </>
                      )}
                      <Button
                        onClick={() => window.open(previewUrl, '_blank', 'noopener,noreferrer')}
                        className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 flex items-center gap-2 mx-auto"
                      >
                        <ExternalLink className="w-4 h-4" />
                        Open PDF in New Tab
                      </Button>
                      <p className="text-sm text-gray-500">
                        {permissionDenied
                          ? 'New tab will work regardless of iframe settings'
                          : 'The PDF will open in a new browser tab with full functionality'
                        }
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer with close button */}
        <div className="p-4 border-t bg-gray-50 rounded-b-lg">
          <div className="flex justify-between items-center">
            {previewUrl && !loading && !error && (
              <Button
                onClick={() => window.open(previewUrl, '_blank', 'noopener,noreferrer')}
                variant="outline"
                className="flex items-center gap-2"
              >
                <ExternalLink className="w-4 h-4" />
                Open in New Tab
              </Button>
            )}
            <Button onClick={onClose} variant="default" className="ml-auto">
              Close Preview
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FilePreviewModal;
