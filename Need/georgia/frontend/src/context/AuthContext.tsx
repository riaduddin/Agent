"use client";

import React, {
  createContext,
  useState,
  useContext,
  ReactNode,
  useEffect,
  useCallback,
  useMemo,
} from "react";
import { registerSessionExpiredCallback } from "@/lib/authEvents";
import axios from "axios";
import { useRouter } from "next/navigation";

// --- Interfaces ---
interface Application {
  id: number;
  name: string;
  description: string;
  url: string;
  status: string;
  user_count: number;
  created_date: string;
  last_updated: string;
}

interface FileCategory {
  id: string;
  code: string;
  name: string;
  description: string;
  status: "active" | "inactive";
  user_count: number;
  created_date: string;
  last_updated: string;
}

interface FileManagementPermissions {
  can_rename_source: boolean;
  can_delete_source: boolean;
  can_upload: boolean;
  can_create_root_folder_source: boolean;
  can_create_folder_source: boolean;
  can_delete_destination: boolean;
  can_create_root_folder_destination: boolean;
  can_create_folder_destination: boolean;
  can_transfer: boolean;
  can_view_all_transfer_history: boolean;
}

interface AdminPanelAccessPermission {
  can_access_regions: boolean;
  can_manage_roles: boolean;
  can_manage_users: boolean;
  can_update_settings: boolean;
}

interface User {
  id: string;
  email: string;
  first_name: string;
  last_name: string;
  role: string;
  status: string;
  last_login: string;
  created_date: string;
  assigned_applications: Application[];
  assigned_file_categories: FileCategory[];
  file_management_permissions?: FileManagementPermissions;
  admin_panel_access_permission?: AdminPanelAccessPermission;
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
interface TreeViewItem {
  id: string;
  name: string;
  path: string;
  type: "Folder" | "File";
  created_date: string | null;
  children: TreeViewItem[];
}

interface TokenInfo {
  user_id: string;
  role: string;
  token_type: string;
  issued_at: string;
  expires_at: string;
}

interface VerifyResponse {
  valid: boolean;
  user: User;
  token_info: TokenInfo;
  access_token: string;
  refresh_token: string;
}

interface AuthContextType {
  isSessionExpired: boolean;
  showSessionExpiredDialog: () => void;
  hideSessionExpiredDialog: () => void;
  loading: boolean;
  user: User | null;
  setUser: (user: User | null) => void;
  tokenInfo: TokenInfo | null;
  logout: () => void;
}

// --- Context Creation ---
export const AuthContext = createContext<AuthContextType | undefined>(
  undefined
);

// --- Provider ---
export const AuthProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [isSessionExpired, setIsSessionExpired] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [tokenInfo, setTokenInfo] = useState<TokenInfo | null>(null);
  const router = useRouter();

  // --- Actions ---

  const showSessionExpiredDialog = useCallback(() => {
    if (!isSessionExpired) {
      setIsSessionExpired(true);
    }
  }, [isSessionExpired]);

  const hideSessionExpiredDialog = useCallback(() => {
    setIsSessionExpired(false);
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    localStorage.clear();
    window.location.href = `${process.env.NEXT_PUBLIC_HOST_URL}/admin/logout`;
  }, []);

  // --- Effects ---

  // 1. Listen for Axios "Session Expired" events
  useEffect(() => {
    const cleanup = registerSessionExpiredCallback(() => {
      showSessionExpiredDialog();
    });
    return cleanup;
  }, [showSessionExpiredDialog]);

  // 2. Initialize Session
  useEffect(() => {
    const initializeSession = async () => {
      try {
        const searchParams = new URLSearchParams(window.location.search);

        // Priority: 1. URL Param (Link click), 2. LocalStorage (Reload)
        const urlAccessToken = searchParams.get("accessToken");
        const storedAccessToken = localStorage.getItem("access_token");

        const tokenToVerify = urlAccessToken || storedAccessToken;

        if (!tokenToVerify) {
          throw new Error("No access token found");
        }

        const { data } = await axios.get<VerifyResponse>(
          `${process.env.NEXT_PUBLIC_ADMIN_API_URL}/auth/verify?token=${tokenToVerify}`
        );

        if (data && data.valid) {
          setUser({
            ...data.user,
            assigned_applications: data.user?.assigned_applications
              ?.filter(
                (app) => app.status.toLowerCase() === "active"
              ) || [],
            assigned_file_categories: data.user?.assigned_file_categories?.filter(
              (category) => category.status === "active"
            ) || [],
          });
          setTokenInfo(data.token_info);

          // Update Storage with latest tokens from verify response
          localStorage.setItem(
            "access_token",
            data.access_token || tokenToVerify
          );
          if (data.refresh_token) {
            localStorage.setItem("refresh_token", data.refresh_token);
          }

          // If we came from a magic link, clean the URL so the user doesn't share the token
          if (urlAccessToken) {
            router.replace("/chat");
          }
        } else {
          throw new Error("Token validation failed");
        }
      } catch (error) {
        console.error("Session initialization failed:", error);
        // Do not auto-redirect here to avoid redirect loops on public pages.
        // Instead, just ensure user is null and loading is done.
        setUser(null);
        logout();
      } finally {
        setLoading(false);
      }
    };

    initializeSession();
  }, [logout, router]);

  // Memoize context value to prevent unnecessary re-renders in children
  const contextValue = useMemo(
    () => ({
      isSessionExpired,
      showSessionExpiredDialog,
      hideSessionExpiredDialog,
      user,
      setUser,
      loading,
      tokenInfo,
      logout,
    }),
    [
      isSessionExpired,
      showSessionExpiredDialog,
      hideSessionExpiredDialog,
      user,
      loading,
      tokenInfo,
      logout,
    ]
  );

  return (
    <AuthContext.Provider value={contextValue}>{children}</AuthContext.Provider>
  );
};

// --- Hook ---
export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};
