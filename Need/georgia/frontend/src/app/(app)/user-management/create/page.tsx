"use client";

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useMutation } from '@tanstack/react-query';
import axiosInstance from '@/lib/axiosInstance';
import { AxiosError } from 'axios'; // Import AxiosError
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import Link from 'next/link';
import { useToast } from '@/components/ui/use-toast'; // Assuming toast component is available
import { Eye, EyeOff } from 'lucide-react'; // Import eye icons
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"; // Import Select components

interface CreateUserData {
  name: string; // Use name instead of username
  email: string;
  password?: string; // Password can be optional for updates, but required for creation
  role?: string; // Add role field
}

interface ErrorResponse {
  message: string;
}

const createUser = async (userData: CreateUserData): Promise<void> => {
  if (userData.role === undefined) {
    userData.role = 'user'; // Default to 'user' if no role is provided
  }
  await axiosInstance.post('/users', userData); // Corrected endpoint - Removed /backend/api/v1
};

const CreateUserPage: React.FC = () => {
  const router = useRouter();
  const { toast } = useToast(); // Get the toast function from the hook
  const [name, setName] = useState(''); // Change state variable to name
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('user'); // State for selected role, default to 'user'
  const [errorMessage, setErrorMessage] = useState<string | null>(null); // State for error message
  const [showPassword, setShowPassword] = useState(false); // State for password visibility

  const createMutation = useMutation({
    mutationFn: createUser,
    onSuccess: () => {
      setErrorMessage(null); // Clear any previous error message
      toast({
        title: "User Created",
        description: "New user has been successfully created.",
      });
      router.push('/user-management'); // Redirect to user list page
    },
    onError: (error: AxiosError<ErrorResponse>) => {
      // Attempt to extract a specific error message from the backend response
      const backendErrorMessage = error.response?.data?.message || error.message;
      setErrorMessage(backendErrorMessage); // Set the error message state
      // Optionally still show a toast for consistency, or remove if only showing on form
      toast({
        title: "Error Creating User",
        description: `Failed to create user: ${backendErrorMessage}`,
        variant: "destructive",
      });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null); // Clear error message on new submission
    createMutation.mutate({ name, email, password, role }); // Include role in mutation data
  };

  return (
    <div className="container mx-auto py-8">
      {/* Container for title and back button */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Create New User</h1>
        <Button variant="outline" size="sm" asChild>
          <Link href="/user-management">Back to Users</Link>
        </Button>
      </div>

      {/* Display error message above the form */}
      {errorMessage && (
        <div className="mb-4 text-red-500">{errorMessage}</div>
      )}

      <form onSubmit={handleSubmit}>
        <div className="grid gap-4 py-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="name" className="text-right"> {/* Change label to Name */}
              Name
            </Label>
            <Input
              id="name" // Change id to name
              value={name} // Bind to name state
              onChange={(e) => setName(e.target.value)} // Update name state
              className="col-span-3 border border-gray-300" // Add border classes
              required
            />
          </div>
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="email" className="text-right">
              Email
            </Label>
            <Input
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="col-span-3 border border-gray-300" // Add border classes
              type="email"
              required
            />
          </div>
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="password" className="text-right">
              Password
            </Label>
            <div className="col-span-3 relative"> {/* Add relative container */}
              <Input
                id="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                type={showPassword ? "text" : "password"} // Conditionally set type
                className="pr-10 border border-gray-300" // Add border classes and padding
                required
              />
              <Button
                type="button" // Prevent form submission
                variant="ghost"
                size="sm"
                className="absolute right-0 top-0 h-full px-3 py-1" // Position icon button
                onClick={() => setShowPassword(!showPassword)} // Toggle password visibility
              >
                {showPassword ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>
          {/* Role selection dropdown */}
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="role" className="text-right">
              Role
            </Label>
            <Select onValueChange={setRole} defaultValue={role}>
              <SelectTrigger id="role" className="col-span-3">
                <SelectValue placeholder="Select a role" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="user">User</SelectItem>
                <SelectItem value="admin">Admin</SelectItem>
                <SelectItem value="manager">Manager</SelectItem>
                <SelectItem value="super-user">Super User</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {/* Add other fields as needed */}
        </div>

        <div className="flex justify-end gap-4">
          <Button variant="outline" asChild>
            <Link href="/user-management">Cancel</Link>
          </Button>
          <Button type="submit" disabled={createMutation.isPending}>
            {createMutation.isPending ? 'Creating...' : 'Create User'}
          </Button>
        </div>
      </form>
    </div>
  );
};

export default CreateUserPage;
