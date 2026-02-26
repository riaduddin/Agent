"use client";

import React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import axiosInstance from '@/lib/axiosInstance';
import { isAxiosError } from 'axios'; // Import isAxiosError
import { useAuth } from '@/context/AuthContext';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import Link from 'next/link';
import { toast } from '@/components/ui/use-toast'; // Assuming toast component is available
import { Loader2, UserPlus, Mail, Edit2, Trash2 } from 'lucide-react';
import { cn } from "@/lib/utils";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"; // Import AlertDialog components

interface User {
  id: string;
  name: string; // Use name instead of username
  email: string;
  role?: string; // Add role field
  // Add other user properties as needed
}

const fetchUsers = async (): Promise<User[]> => {
  const response = await axiosInstance.get('/users'); // Corrected endpoint - Removed /backend/api/v1
  return response.data;
};

const deleteUser = async (userId: string): Promise<void> => {
  await axiosInstance.delete(`/users/${userId}`); // Corrected endpoint - Removed /backend/api/v1
};

const UserManagementPage: React.FC = () => {
  const { user } = useAuth(); // Use the custom hook instead of useContext(AuthContext)
  const queryClient = useQueryClient();
  const { data: users, isLoading, error, } = useQuery<User[]>({
    queryKey: ['users'],
    queryFn: fetchUsers,
  });


  const deleteMutation = useMutation({
    mutationFn: deleteUser,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] }); // Refetch users after deletion
      toast({
        title: "User Deleted",
        description: "User has been successfully deleted.",
      });
    },
    onError: (error) => {
      toast({
        title: "Error Deleting User",
        description: `Failed to delete user: ${(error as Error).message}`,
        variant: "destructive",
      });
    },
  });

  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = React.useState(false);
  const [userToDeleteId, setUserToDeleteId] = React.useState<string | null>(null);
  const [deletingUserId, setDeletingUserId] = React.useState<string | null>(null); // State to track the user being deleted

  const handleDeleteUser = (userId: string) => {
    setUserToDeleteId(userId);
    setIsDeleteDialogOpen(true);
  };

  const confirmDelete = () => {
    if (userToDeleteId) {
      setDeletingUserId(userToDeleteId); // Set the user being deleted
      deleteMutation.mutate(userToDeleteId, {
        onSuccess: () => {
          queryClient.invalidateQueries({ queryKey: ['users'] }); // Refetch users after deletion
          toast({
            title: "User Deleted",
            description: "User has been successfully deleted.",
          });
          setDeletingUserId(null); // Clear deleting user on success
        },
        onError: (error) => {
          toast({
            title: "Error Deleting User",
            description: `Failed to delete user: ${(error as Error).message}`,
            variant: "destructive",
          });
          setDeletingUserId(null); // Clear deleting user on error
        },
      });
      setUserToDeleteId(null); // Clear the user ID after initiating deletion
    }
    setIsDeleteDialogOpen(false); // Close the dialog
  };

  if (isLoading) {
    return (
      <div className="container mx-auto py-8 flex justify-center items-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" /> {/* Use a slightly larger spinner */}
      </div>
    );
  }

  // Check for 403 Forbidden error specifically
  if (error) {
    if (isAxiosError(error) && error.response?.status === 403) {
      // Render Permission Denied message for 403 errors
      return (
        <div className="container mx-auto py-8">
          <div className="flex items-center justify-center h-full">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
              <h2 className="text-xl font-bold text-red-600 dark:text-red-400">Permission Denied</h2>
              <p className="mt-2 text-gray-700 dark:text-gray-300">You do not have permission to view this page.</p>
            </div>
          </div>
        </div>
      );
    }
    // Render generic error message for other errors
    return <div className="container mx-auto py-8 text-red-500">Error loading users: {(error as Error).message}</div>;
  }

  // This check for user?.role === 'user' might become redundant if the backend
  // consistently returns 403 for unauthorized users, but keeping it for
  // robustness in case the backend check is bypassed or for initial loading state.
  if (user?.role === 'user') {
    return (
      <div className="container mx-auto py-8">
        <div className="flex items-center justify-center h-full">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-bold text-red-600 dark:text-red-400">Permission Denied</h2>
            <p className="mt-2 text-gray-700 dark:text-gray-300">You do not have permission to view this page.</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-[#F8F8F8] p-8 overflow-y-auto">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header Section */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <h1
              className="text-[28px] font-bold text-[#255c5d] leading-tight"
              style={{ fontFamily: 'var(--unnamed-font-family-georgia)' }}
            >
              User Management
            </h1>
            <p
              className="text-xs text-[#707070] mt-1"
              style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}
            >
              Manage system users, their roles and access permissions
            </p>
          </div>
          <Button asChild className="bg-[#255c5d] hover:bg-[#1E4E55] text-white px-6 h-10 rounded-[4px] uppercase text-[10px] font-bold tracking-widest transition-colors">
            <Link href="/user-management/create">
              <UserPlus className="mr-2 h-4 w-4" />
              Create New User
            </Link>
          </Button>
        </div>

        {/* Users Table Card */}
        <div className="bg-white border border-[#E0E0E0] rounded-[8px] overflow-hidden shadow-sm">
          {/* Search/Filter Bar (Optional, can add later) */}
          <div className="bg-[#F1F8F8] px-6 py-4 border-b border-[#E0E0E0]">
            <h2 className="text-[11px] font-bold text-[#255c5d] uppercase tracking-[0.2em]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>Active System Users</h2>
          </div>

          <div className="overflow-x-auto">
            <Table>
              <TableHeader className="bg-gray-50/50">
                <TableRow className="border-b border-[#E0E0E0] hover:bg-transparent">
                  <TableHead className="text-[11px] font-bold text-[#707070] uppercase tracking-wider py-4 pl-6">Full Name</TableHead>
                  <TableHead className="text-[11px] font-bold text-[#707070] uppercase tracking-wider py-4">Email Address</TableHead>
                  <TableHead className="text-[11px] font-bold text-[#707070] uppercase tracking-wider py-4">System Role</TableHead>
                  <TableHead className="text-[11px] font-bold text-[#707070] uppercase tracking-wider py-4 text-right pr-6">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {users?.map((u) => (
                  <TableRow key={u.id} className="border-b border-[#E0E0E0] hover:bg-gray-50 transition-colors">
                    <TableCell className="py-4 pl-6">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-full bg-[#E8F3F3] flex items-center justify-center text-[#255c5d] font-bold text-xs uppercase">
                          {u.name.substring(0, 1)}
                        </div>
                        <span className="text-sm font-medium text-[#000000]">{u.name}</span>
                      </div>
                    </TableCell>
                    <TableCell className="py-4">
                      <div className="flex items-center gap-2 text-sm text-[#707070]">
                        <Mail className="h-3 w-3 opacity-50" />
                        {u.email}
                      </div>
                    </TableCell>
                    <TableCell className="py-4">
                      <div className="flex items-center gap-2">
                        <span className={cn(
                          "text-[10px] font-bold uppercase tracking-tight px-2 py-0.5 rounded-full border",
                          u.role === 'superadmin'
                            ? "bg-[#255c5d] text-white border-[#255c5d]"
                            : "bg-[#E8F3F3] text-[#255c5d] border-[#255c5d]/20"
                        )}>
                          {u.role}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell className="py-4 text-right pr-6">
                      <div className="flex items-center justify-end gap-3">
                        <Link
                          href={`/user-management/edit/${u.id}`}
                          className="p-1.5 text-[#255c5d] hover:bg-[#E8F3F3] rounded transition-colors"
                          title="Edit User"
                        >
                          <Edit2 className="h-4 w-4" />
                        </Link>
                        <button
                          onClick={() => handleDeleteUser(u.id)}
                          disabled={deleteMutation.isPending && deletingUserId === u.id}
                          className="p-1.5 text-red-600 hover:bg-red-50 rounded transition-colors disabled:opacity-50"
                          title="Delete User"
                        >
                          {deleteMutation.isPending && deletingUserId === u.id ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Trash2 className="h-4 w-4" />
                          )}
                        </button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>

        {/* AlertDialog for delete confirmation */}
        <AlertDialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
          <AlertDialogContent className="rounded-[8px]">
            <AlertDialogHeader>
              <AlertDialogTitle className="text-[#255c5d]" style={{ fontFamily: 'var(--unnamed-font-family-georgia)' }}>Delete User Account</AlertDialogTitle>
              <AlertDialogDescription className="text-sm text-[#707070]" style={{ fontFamily: 'var(--unnamed-font-family-montserrat)' }}>
                Are you sure you want to delete this user? This action is permanent and will revoke all access for <strong>{users?.find(u => u.id === userToDeleteId)?.name}</strong>.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel className="rounded-[4px] border-[#E0E0E0] text-[12px] uppercase font-bold tracking-widest">Cancel</AlertDialogCancel>
              <AlertDialogAction
                onClick={confirmDelete}
                className="bg-red-600 hover:bg-red-700 text-white rounded-[4px] text-[12px] uppercase font-bold tracking-widest transition-colors"
              >
                Delete User
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </div>
  );
};

export default UserManagementPage;
