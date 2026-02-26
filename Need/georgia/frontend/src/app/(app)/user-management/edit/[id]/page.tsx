"use client";

import React, { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { useQuery, useMutation } from '@tanstack/react-query';
import axiosInstance from '@/lib/axiosInstance';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import Link from 'next/link';
import { useToast } from '@/components/ui/use-toast'; // Assuming toast component is available
import {
  Select as ShadSelect,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"; // Import Select components
import Select, { OnChangeValue, MultiValue } from 'react-select';
import makeAnimated from 'react-select/animated';

const animatedComponents = makeAnimated();
interface OptionType {
  label: string;
  value: string;
}

interface UserData {
  name: string; // Use name instead of username
  email: string;
  password?: string;
  role?: string; // Add role field
  accessible_categories?: string[];
}

const fetchUser = async (userId: string): Promise<UserData> => {
  const response = await axiosInstance.get(`/users/${userId}`); // Corrected endpoint - Removed /backend/api/v1
  return response.data;
};

const fetchCategories = async (): Promise<string[]> => {
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_API_URL;
    const response = await axiosInstance.get(`${backendUrl}/backend/api/v2/system/categories`);
    return response.data;
};

const updateUser = async ({ userId, userData }: { userId: string; userData: UserData }): Promise<void> => {
  await axiosInstance.put(`/users/${userId}`, userData); // Corrected endpoint - Removed /backend/api/v1
};

const EditUserPage: React.FC = () => {
  const params = useParams();
  const router = useRouter();
  const userId = params.id as string; // Access the user ID from the route parameters

  const { data: user, isLoading, error } = useQuery<UserData>({
    queryKey: ['user', userId],
    queryFn: () => fetchUser(userId),
    enabled: !!userId, // Only fetch if userId is available
  });

  const [name, setName] = useState(''); // Change state variable to name
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState(''); // State for password, not pre-filled
  const [role, setRole] = useState('user'); // State for selected role, default to 'user'
  const [accessibleCategories, setAccessibleCategories] = useState<string[]>([]);

  const { data: allCategories, isLoading: isLoadingCategories } = useQuery<string[]>({
    queryKey: ['allCategories'],
    queryFn: fetchCategories,
  });

  // Effect to pre-populate form when user data is loaded
  useEffect(() => {
    if (user) {
      setName(user.name); // Pre-populate name
      setEmail(user.email);
      setRole(user.role || 'user'); // Pre-populate role, default to 'user'
      setAccessibleCategories(user.accessible_categories || []);
  // Do NOT pre-fill password for security reasons
    }
  }, [user]);

  const { toast } = useToast(); // Get the toast function from the hook

  const updateMutation = useMutation({
    mutationFn: updateUser,
    onSuccess: () => {
      toast({
        title: "User Updated",
        description: "User information has been successfully updated.",
      });
      router.push('/user-management'); // Redirect to user list page
    },
    onError: (error) => {
      toast({
        title: "Error Updating User",
        description: `Failed to update user: ${(error as Error).message}`,
        variant: "destructive",
      });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const updateData: UserData = { name, email, role, accessible_categories: accessibleCategories }; // Include name and role in update data
    if (password) { // Only include password in update if it's not empty
      updateData.password = password;
    }
    updateMutation.mutate({ userId, userData: updateData });
  };

  if (isLoading) {
    return <div className="container mx-auto py-8">Loading user data...</div>;
  }

  if (error) {
    return <div className="container mx-auto py-8 text-red-500">Error loading user data: {(error as Error).message}</div>;
  }

  if (!user) {
      return <div className="container mx-auto py-8">User not found.</div>;
  }


  return (
    <div className="container mx-auto py-8">
      {/* Container for title and back button */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Edit User: {user.name}</h1> {/* Display name */}
        <Button variant="outline" size="sm" asChild>
          <Link href="/user-management">Back to Users</Link>
        </Button>
      </div>

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
              className="col-span-3"
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
              className="col-span-3"
              type="email"
              required
            />
          </div>
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="password" className="text-right">
              Password (Leave blank to keep current)
            </Label>
            <Input
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="col-span-3"
              type="password"
              placeholder="Leave blank to keep current password"
            />
          </div>
          {/* Add other fields as needed */}
          {/* Role selection dropdown */}
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="role" className="text-right">
              Role
            </Label>
            <ShadSelect onValueChange={setRole} value={role}> {/* Use value for pre-selection */}
              <SelectTrigger id="role" className="col-span-3">
                <SelectValue placeholder="Select a role" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="user">User</SelectItem>
                <SelectItem value="admin">Admin</SelectItem>
                <SelectItem value="manager">Manager</SelectItem>
                <SelectItem value="super-user">Super User</SelectItem>                
              </SelectContent>
            </ShadSelect>
          </div>
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="categories" className="text-right">
              Accessible Categories
            </Label>
            <Select
                id="categories"
                closeMenuOnSelect={false}
                components={animatedComponents}
                isMulti
                options={(allCategories || []).map(cat => ({ label: cat, value: cat }))}
                className="col-span-3"
                value={(allCategories || [])
                    .filter(cat => accessibleCategories.includes(cat))
                    .map(cat => ({ label: cat, value: cat }))
                }
                onChange={(selectedOptions) => {
                    setAccessibleCategories(selectedOptions ? (selectedOptions as OptionType[]).map(option => option.value) : []);
                }}
                isLoading={isLoadingCategories}
            />
          </div>
        </div>

        <div className="flex justify-end gap-4">
          <Button variant="outline" asChild>
            <Link href="/user-management">Cancel</Link>
          </Button>
          <Button type="submit" disabled={updateMutation.isPending}>
            {updateMutation.isPending ? 'Saving...' : 'Save Changes'}
          </Button>
        </div>
      </form>
    </div>
  );
};

export default EditUserPage;
