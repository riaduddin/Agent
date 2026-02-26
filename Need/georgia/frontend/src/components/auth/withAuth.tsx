// src/components/auth/withAuth.tsx
"use client"; // Needed for hooks like useRouter, useEffect

import { useRouter } from 'next/navigation';
import React, { useEffect, useState, ComponentType } from 'react';

// Removed unused WithAuthProps interface

const withAuth = <P extends object>(WrappedComponent: ComponentType<P>) => { // Changed constraint to object
  const AuthComponent: React.FC<P> = (props) => {
    const router = useRouter();
    // Add a loading state to prevent rendering mismatch during initial check
    const [isLoading, setIsLoading] = useState(true);
    const [isAuthenticated, setIsAuthenticated] = useState(false);


    useEffect(() => {
      // Perform the check only on the client-side after mount
      const token = localStorage.getItem('access_token');
      if (!token) {
        router.replace('/login');
      } else {
        // Token exists, allow rendering the component
        setIsAuthenticated(true);
      }
      setIsLoading(false); // Mark loading as complete
    }, [router]);

    // While loading, render null or a loading indicator to match server (which also wouldn't have run useEffect)
    if (isLoading) {
      // You could return a proper loading spinner component here
      return null; // Or <LoadingSpinner />;
    }

    // If authenticated after check, render the wrapped component
    if (isAuthenticated) {
      return <WrappedComponent {...props} />;
    }

    // If not authenticated after check (and redirect hasn't happened yet), render null
    return null;
  };

  // Set display name for easier debugging
  AuthComponent.displayName = `WithAuth(${WrappedComponent.displayName || WrappedComponent.name || 'Component'})`;

  return AuthComponent;
};

export default withAuth;
