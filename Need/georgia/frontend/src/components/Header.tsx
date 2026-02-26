/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/context/AuthContext";
import { useFilePermissions } from "@/hooks/use-file-permissions";
import axiosInstance from "@/lib/axiosInstance";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  User,
  LogOut,
  Wrench,
  Settings,
  File,
  Building2,
  ChevronDown,
  ArrowLeft,
  AlertCircle,
  Menu,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

// Helper function to generate initials from first and last name
const generateInitials = (firstName: string, lastName: string): string => {
  const first = firstName?.trim() || "";
  const last = lastName?.trim() || "";

  if (!first && !last) return "";
  if (!last) return first.charAt(0).toUpperCase();

  return first.charAt(0).toUpperCase() + last.charAt(0).toUpperCase();
};

// Avatar component to display initials
const AvatarPlaceholder = ({
  firstName,
  lastName,
}: {
  firstName: string;
  lastName: string;
}) => {
  const initials = generateInitials(firstName, lastName);
  return (
    <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center text-xs text-gray-700 font-semibold">
      {initials}
    </div>
  );
};

const MenuIcon = ({ className }: { className?: string }) => (
  <span className={`inline-block w-6 h-6 ${className}`}>☰</span>
);

// Define props interface
interface HeaderProps {
  toggleSidebar: () => void;
  isSidebarOpen: boolean;
}

interface UserInfo {
  name: string;
  email: string;
  role?: string;
  assigned_applications?: AssignedApplication[];
}

interface AssignedApplication {
  id: number;
  name: string;
  description: string;
  url: string;
  status: string;
  user_count: number;
  created_date: string;
  last_updated: string;
}

const Header: React.FC<HeaderProps> = ({ toggleSidebar, isSidebarOpen }) => {
  const router = useRouter();
  const { user } = useAuth();
  const { canAccessAdminPanel } = useFilePermissions();
  const [currentRegion, setCurrentRegion] = useState<string>("Region 14");
  const [showLogoutDialog, setShowLogoutDialog] = useState(false);

  // Fix: Force restore body interaction if a modal gets stuck
  useEffect(() => {
    if (!showLogoutDialog) {
      // Small delay to allow Radix UI to finish its own transitions
      const timer = setTimeout(() => {
        document.body.style.pointerEvents = "";
        document.body.style.overflow = "";
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [showLogoutDialog]);

  // Get full name from user object
  const fullName =
    user?.first_name && user?.last_name
      ? `${user.first_name} ${user.last_name}`.trim()
      : user?.first_name || user?.last_name || "User";

  // Effect to detect current region from URL
  useEffect(() => {
    const path = window.location.pathname;
    const regionMatch = path.match(/\/(r\d+)/);
    if (regionMatch) {
      const regionSlug = regionMatch[1];
      const matchedApp = user?.assigned_applications?.find((app) =>
        app.url.includes(`/${regionSlug}`)
      );
      if (matchedApp) {
        setCurrentRegion(matchedApp.name);
      }
    }
  }, [user?.assigned_applications]);

  const handleLogout = () => {
    setShowLogoutDialog(true);
  };

  const confirmLogout = async () => {
    try {
      await axiosInstance.post("/auth/logout");
    } catch (error) {
      console.error("Error calling logout API:", error);
    }

    localStorage.removeItem("access_token");
    localStorage.removeItem("user_info");
    localStorage.clear();
    window.location.href = `${process.env.NEXT_PUBLIC_HOST_URL}/admin/logout`;
  };

  const handleApplicationSwitch = (app: AssignedApplication) => {
    if (app.url) {
      // Get access token from localStorage
      const accessToken = localStorage.getItem("access_token");

      // Construct URL with access token
      let targetUrl = app.url;
      if (accessToken) {
        const separator = app.url.includes("?") ? "&" : "?";
        targetUrl = `${app.url}${separator}accessToken=${encodeURIComponent(
          accessToken
        )}`;
      }

      // If it's an external URL, navigate to it
      if (app.url.startsWith("http")) {
        window.location.href = targetUrl;
      } else {
        // If it's an internal route, use Next.js router
        router.push(targetUrl);
      }
      setCurrentRegion(app.name);
    }
  };

  function truncateText(text: any, maxLength = 100) {
    if (typeof text !== "string") return "";
    if (text.length <= maxLength) {
      return text;
    }
    return text.substring(0, maxLength).trim() + "...";
  }

  function capitalizeFirstWord(text: string): string {
    if (!text) return "";
    const words = text.split("-");
    const firstWord = words[0].charAt(0).toUpperCase() + words[0].slice(1);
    const restOfWords = words.slice(1).join(" ");
    return firstWord + (restOfWords ? " " + restOfWords : "");
  }

  const handleBackToAdmin = () => {
    window.location.href = `${process.env.NEXT_PUBLIC_HOST_URL}/admin/dashboard`;
  };

  return (
    <>
      <header className="h-16 bg-[#FFFFFF] border-b border-[#E0E0E0] flex items-center justify-between px-8 z-50">
        {/* Left side: Logo/Title */}
        <div className="flex items-center gap-4">
          <button
            onClick={toggleSidebar}
            className="text-[#707070] hover:text-black md:hidden focus:outline-none transition-colors"
          >
            <Menu className="w-6 h-6" />
          </button>
          <span className="text-[20px] font-bold tracking-tight text-[#000000]" style={{ fontFamily: 'var(--unnamed-font-family-georgia)' }}>
            IntelliDocFinder
          </span>
        </div>

        {/* Right side: Region Identifier + User Icon */}
        <div className="flex items-center gap-6">
          {/* Back to Admin Button */}
          {canAccessAdminPanel && (
            <button
              onClick={handleBackToAdmin}
              className="flex items-center bg-white border border-[#E0E0E0] rounded-[4px] h-[36px] px-3 gap-2 cursor-pointer hover:bg-[#F8F8F8] transition-colors shadow-sm"
            >
              <ArrowLeft className="h-4 w-4 text-[#707070]" />
              <span className="text-[13px] font-medium text-[#000000]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>
                Back to Admin
              </span>
            </button>
          )}

          {/* Region Selector Dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger className="focus:outline-none">
              <div className="flex items-center bg-white border border-[#E0E0E0] rounded-[4px] h-[36px] px-3 gap-2 cursor-pointer hover:bg-[#F8F8F8] transition-colors shadow-sm">
                <Building2 className="h-4 w-4 text-[#707070]" />
                <span className="text-[13px] font-medium text-[#000000]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>
                  {currentRegion}
                </span>
                <ChevronDown className="h-3 w-3 text-[#707070]" />
              </div>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-56 mt-2" align="end">
              <DropdownMenuLabel className="text-xs font-medium text-gray-500 px-2 py-1.5">
                Switch Region
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <div className="max-h-64 overflow-y-auto p-1">
                {user?.assigned_applications && user.assigned_applications.length > 0 ? (
                  user.assigned_applications.map((app) => (
                    <DropdownMenuItem
                      key={app.id}
                      onClick={() => handleApplicationSwitch(app)}
                      className="flex items-center cursor-pointer p-2 rounded hover:bg-[#EEF4F4] transition-colors"
                    >
                      <Building2 className="mr-2 h-4 w-4 text-[#707070]" />
                      <span className="text-[13px] text-[#000000]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>
                        {app.name}
                      </span>
                      {currentRegion === app.name && (
                        <span className="ml-auto w-2 h-2 rounded-full bg-[#255c5d]" />
                      )}
                    </DropdownMenuItem>
                  ))
                ) : (
                  <div className="text-xs text-gray-500 p-2 text-center">No regions available</div>
                )}
              </div>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* User Avatar */}
          <DropdownMenu>
            <DropdownMenuTrigger className="focus:outline-none">
              <div className="w-10 h-10 rounded-full bg-[#E0E0E0] flex items-center justify-center text-[#000000] cursor-pointer hover:bg-[#D0D0D0] transition-all shadow-sm active:scale-95 font-medium text-[13px]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>
                {generateInitials(user?.first_name || "", user?.last_name || "")}
              </div>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-64 mt-2" align="end" forceMount>
              <DropdownMenuLabel className="font-normal p-4 bg-[#F8F8F8]">
                <div className="flex flex-col space-y-1">
                  <p className="text-[14px] font-bold text-[#000000]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>{fullName}</p>
                  <p className="text-[11px] text-[#707070] truncate" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>{user?.email || "No email"}</p>
                  <div className="pt-2">
                    <Badge variant="secondary" className="bg-[#B4C9BF] text-[#255c5d] hover:bg-[#B4C9BF] font-bold text-[10px] px-2 py-0 capitalize">
                      {user?.role?.replace(/_/g, " ") || "User"}
                    </Badge>
                  </div>
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <div className="p-2">
                <DropdownMenuItem asChild>
                  <Link href="/profile" className="flex items-center cursor-pointer p-2 rounded hover:bg-[#EEF4F4] transition-colors group">
                    <User className="mr-3 h-4 w-4 text-[#707070] group-hover:text-[#255c5d]" />
                    <span className="text-[13px] text-[#000000]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Profile</span>
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/initial-setup-statistics" className="flex items-center cursor-pointer p-2 rounded hover:bg-[#EEF4F4] transition-colors group">
                    <Settings className="mr-3 h-4 w-4 text-[#707070] group-hover:text-[#255c5d]" />
                    <span className="text-[13px] text-[#000000]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Setup Documents</span>
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/diagnosis" className="flex items-center cursor-pointer p-2 rounded hover:bg-[#EEF4F4] transition-colors group">
                    <Wrench className="mr-3 h-4 w-4 text-[#707070] group-hover:text-[#255c5d]" />
                    <span className="text-[13px] text-[#000000]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Diagnosis</span>
                  </Link>
                </DropdownMenuItem>
              </div>
              <DropdownMenuSeparator />
              <div className="p-2">
                <DropdownMenuItem onClick={handleLogout} className="flex items-center cursor-pointer p-2 rounded transition-colors group">
                  <LogOut className="mr-3 h-4 w-4" />
                  <span className="text-[13px] font-bold" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Log out</span>
                </DropdownMenuItem>
              </div>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      <AlertDialog open={showLogoutDialog} onOpenChange={setShowLogoutDialog}>
        <AlertDialogContent className="bg-white p-6 max-w-[450px] rounded-sm border-none shadow-2xl">
          {/* Close button in top right */}
          <button
            onClick={() => setShowLogoutDialog(false)}
            className="absolute top-4 right-4 text-black hover:opacity-70 transition-opacity"
          >
            <X className="h-5 w-5" />
          </button>

          <AlertDialogHeader className="flex flex-row items-start gap-3 space-y-0">
            <div className="flex flex-col gap-1 text-left">
              <AlertDialogTitle className="text-[18px] font-bold text-black leading-tight flex items-center gap-1" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>
                <AlertCircle className="h-6 w-6 text-[#B01016]" />  Log out
              </AlertDialogTitle>
              <AlertDialogDescription className="text-[14px] text-black font-medium" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>
                Are you sure want to log out?
              </AlertDialogDescription>
            </div>
          </AlertDialogHeader>

          <AlertDialogFooter className="flex flex-row justify-center gap-4 mt-8 sm:justify-center">
            <Button
              variant="outline"
              onClick={confirmLogout}
              className="border-[#255c5d] text-[#255c5d] hover:bg-[#E8F3F3] w-[100px] h-[36px] font-bold text-[12px] uppercase rounded-sm"
            >
              YES
            </Button>
            <Button
              variant="outline"
              onClick={() => setShowLogoutDialog(false)}
              className="border-[#B01016] text-[#000000] hover:bg-red-50 w-[100px] h-[36px] font-bold text-[12px] uppercase rounded-sm"
            >
              NO
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
};

export default Header;
