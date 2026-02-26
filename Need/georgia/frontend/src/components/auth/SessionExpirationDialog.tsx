'use client';

import React from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/context/AuthContext'; // Adjust path if necessary
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'; // Adjust path if necessary

export function SessionExpirationDialog() {
  const { isSessionExpired } = useAuth();
  const router = useRouter();

  const handleLoginRedirect = () => {
    // Clear authentication state (e.g., token in localStorage)
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token'); // Also remove refresh token if used
    // Optionally hide the dialog via context if needed before redirect
    // hideSessionExpiredDialog();
    router.push('/login');
  };

  // Note: The AlertDialog's open state is controlled directly by the `isSessionExpired` prop.
  // We don't need local state here to manage openness.

  return (
    <AlertDialog open={isSessionExpired}>
      {/* No AlertDialogTrigger needed as it's controlled externally */}
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Session Expired</AlertDialogTitle>
          <AlertDialogDescription>
            Your session has expired. Please log in again to continue.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          {/* No Cancel button needed */}
          <AlertDialogAction onClick={handleLoginRedirect}>Login</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
