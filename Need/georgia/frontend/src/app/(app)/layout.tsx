/* eslint-disable @typescript-eslint/no-unused-vars */
// src/app/(app)/layout.tsx - Layout for private routes
"use client";

import { useState } from "react";
import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import React from "react";
import { AuthProvider, useAuth } from "@/context/AuthContext";
import { SessionExpirationDialog } from "@/components/auth/SessionExpirationDialog";
import withAuth from "@/components/auth/withAuth";

import Image from "next/image";

// Loading component to be displayed during auth check
const LoadingSpinner = () => (
  <div className="flex flex-col items-center justify-center h-screen w-screen bg-[#255c5d]">
    <div className="animate-pulse">
      <Image
        src={`${process.env.NEXT_PUBLIC_ASSET_PREFIX}/DHS_White_vert.png`}
        alt="Loading..."
        width={280}
        height={180}
        priority
      />
    </div>
    <p className="text-white text-lg mt-4">Loading, please wait...</p>
  </div>
);

// The actual layout component
const AppLayout = ({ children }: { children: React.ReactNode }) => {
  const { loading } = useAuth();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  if (loading) {
    return <LoadingSpinner />;
  }

  return (
    <>
      <div className="flex flex-col h-screen overflow-hidden bg-white">
        <Header toggleSidebar={toggleSidebar} isSidebarOpen={isSidebarOpen} />
        <div className="flex flex-1 overflow-hidden relative">
          <Sidebar isOpen={isSidebarOpen} toggleSidebar={toggleSidebar} />
          {isSidebarOpen && (
            <div
              onClick={toggleSidebar}
              className="fixed inset-0 bg-black bg-opacity-30 z-30 md:hidden"
            ></div>
          )}
          <main className="flex-1 bg-white relative overflow-y-auto">
            {children}
          </main>
        </div>
      </div>
      <SessionExpirationDialog />
    </>
  );
};

// A new root component that wraps everything with the AuthProvider
const PrivateAppLayout = ({ children }: { children: React.ReactNode }) => {
  return (
    <AuthProvider>
      <AppLayout>{children}</AppLayout>
    </AuthProvider>
  );
};

export default PrivateAppLayout;
