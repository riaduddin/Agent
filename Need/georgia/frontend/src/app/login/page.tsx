/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/no-explicit-any */
"use client"; // Needed for hooks

import React, { useState, useEffect } from "react"; // Import useState
import Image from "next/image";
import { useForm, SubmitHandler } from "react-hook-form";
import { useAuth } from "@/context/AuthContext"; // Import useAuth hook
import { useMutation } from "@tanstack/react-query";
import { isAxiosError } from "axios";
import axiosInstance from "@/lib/axiosInstance";
import { useRouter } from "next/navigation";
// Removed unused Link import
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
// Removed unused Card components as the form is not in a Card anymore
// import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Eye, EyeOff } from "lucide-react"; // Import icons

// Define the type for our form inputs
type LoginFormInputs = {
  email: string;
  password: string;
};

// Define the expected API response structure (adjust as needed)
type LoginApiResponse = {
  access_token: string;
  user: {
    id: string;
    email: string;
    name?: string;
    // Add other user fields returned by backend
  };
  // Add refresh_token if using
};

// Define the mutation function
const loginUser = async (data: LoginFormInputs): Promise<LoginApiResponse> => {
  const response = await axiosInstance.post<LoginApiResponse>(
    "/auth/login",
    data
  );
  return response.data;
};

export default function LoginPage() {
  const { loading } = useAuth();
  const router = useRouter();
  const { setUser, user } = useAuth(); // Use useAuth hook to get setUser
  const [showPassword, setShowPassword] = useState(false); // State for password visibility
  const {
    register,
    handleSubmit,
    formState: { errors },
    setError,
  } = useForm<LoginFormInputs>();

  const mutation = useMutation<LoginApiResponse, Error, LoginFormInputs>({
    mutationFn: loginUser,
    onSuccess: (data) => {
      // Consider using a more secure storage mechanism if available (e.g., httpOnly cookies set by backend)
      localStorage.setItem("access_token", data.access_token);
      localStorage.setItem("user_info", JSON.stringify(data.user));
      setUser(data.user as any); // Set user information in AuthContext
      // Optionally broadcast login event if using cross-tab state management
      // authEvents.dispatchLogin();
      router.push("/chat"); // Redirect to chat page
    },
    onError: (error: unknown) => {
      console.error("Login error:", error);
      let errorMessage = "Login failed. Please try again.";
      if (isAxiosError(error) && error.response?.data?.msg) {
        errorMessage = error.response.data.msg;
      } else if (error instanceof Error) {
        errorMessage = error.message;
      }
      // Display error message associated with the form root or a specific field if applicable
      setError("root.serverError", {
        type: "manual",
        message: errorMessage,
      });
    },
  });

  const onSubmit: SubmitHandler<LoginFormInputs> = (data) => {
    mutation.mutate(data);
  };

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  const [hideManualLogin, setHideManualLogin] = useState(true); // initially true
  useEffect(() => {
    if (typeof window !== "undefined") {
      const hostname = window.location.hostname;
      if (hostname !== "intellidocfinder-dfcsffs.dhs.ga.gov") {
        setHideManualLogin(false); // show manual login
      }

      if (user) {
        router.push("/chat");
      }
    }
  }, []);

  // return (
  //   // Main container: Two-column layout, stacks on small screens
  //   <div className="flex flex-col md:flex-row min-h-screen w-full">
  //     {/* Left Column - 40% width on md+, bg-[#1F2A38] */}
  //     <div className="w-full md:w-4/5 bg-[#1F2A38] flex flex-col items-center justify-center p-8 md:p-12 text-white order-1 md:order-none">
  //       <div className="mb-6 flex flex-col items-center justify-center relative">
  //         {/* Square background with overlay */}
  //         <div className="absolute inset-0 w-100 h-100 bg-gradient-to-br from-teal-500/20 to-blue-600/20 rounded-2xl shadow-2xl backdrop-blur-sm border border-teal-400/30 transform -translate-x-1/2 -translate-y-1/2 top-1/2 left-1/2"></div>
  //         <div className="relative z-10 mb-4">
  //           <Image
  //             src="/r14/DHS_White_vert.png" // Ensure this path is correct
  //             alt="State of Georgia Logo"
  //             width={280} // Reverted to original size
  //             height={180} // Reverted to original size
  //             priority
  //           />
  //         </div>

  //         {/* Text positioned below logo but within the square */}
  //         <div className="relative z-10 text-center top-[-35px] ">
  //           <p className="text-center  text-[#fff] font-bold leading-relaxed max-w-xs">
  //             Division of Family and Children Services Field Fiscal Services
  //           </p>
  //         </div>
  //       </div>
  //     </div>

  //     {/* Right Column - 60% width on md+, bg-[#b6bdc9] */}
  //     <div className="w-full md:w-3/5 flex flex-col items-center justify-center p-8 md:p-12 bg-[#b6bdc9] order-2 md:order-none">
  //       {/* Content wrapper for max-width */}
  //       <div className="w-full max-w-sm">
  //         {/* Reverted title and description */}
  //         <h1 className="text-2xl md:text-3xl font-bold text-[#003057] mb-2 text-center md:text-center">
  //           IntelliDocFinder
  //         </h1>{" "}
  //         {/* Use navy color */}
  //         {hideManualLogin == false && (
  //           <p className="text-sm text-gray-600 mb-8 text-center md:text-left">
  //             Enter your login details below.
  //           </p>
  //         )}
  //         {/* Login Form (No Card Background) */}
  //         <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
  //           {/* Display general server errors */}
  //           {errors.root?.serverError && (
  //             <p className="text-sm text-red-600 text-center">
  //               {errors.root.serverError.message}
  //             </p>
  //           )}
  //           {/* Email Field */}
  //           {hideManualLogin == false && (
  //             <div className="space-y-1">
  //               {/* Reverted label style */}
  //               <Label
  //                 htmlFor="email"
  //                 className="block text-sm font-medium text-gray-700"
  //               >
  //                 {" "}
  //                 {/* Adjusted text color */}
  //                 Email address
  //               </Label>
  //               <Input
  //                 id="email"
  //                 type="email"
  //                 autoComplete="email"
  //                 {...register("email", {
  //                   required: "Email is required",
  //                   pattern: {
  //                     value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
  //                     message: "Invalid email address",
  //                   },
  //                 })}
  //                 // Using simpler border and focus states consistent with shadcn/ui defaults
  //                 className={`mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm ${
  //                   errors.email
  //                     ? "border-red-500 focus:ring-red-500 focus:border-red-500"
  //                     : ""
  //                 }`}
  //                 placeholder="you@example.com"
  //               />
  //               {errors.email && (
  //                 <p className="mt-1 text-xs text-red-600">
  //                   {errors.email.message}
  //                 </p> // Adjusted error style
  //               )}
  //             </div>
  //           )}
  //           {/* Password Field */}
  //           {hideManualLogin == false && (
  //             <div className="space-y-1">
  //               {/* Reverted label style */}
  //               <Label
  //                 htmlFor="password"
  //                 className="block text-sm font-medium text-gray-700"
  //               >
  //                 {" "}
  //                 {/* Adjusted text color */}
  //                 Password
  //               </Label>
  //               <div className="relative mt-1">
  //                 <Input
  //                   id="password"
  //                   type={showPassword ? "text" : "password"}
  //                   autoComplete="current-password"
  //                   {...register("password", {
  //                     required: "Password is required",
  //                   })}
  //                   // Using simpler border and focus states consistent with shadcn/ui defaults
  //                   className={`block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm pr-10 ${
  //                     errors.password
  //                       ? "border-red-500 focus:ring-red-500 focus:border-red-500"
  //                       : ""
  //                   }`}
  //                   placeholder="********"
  //                 />
  //                 {/* Kept password toggle button */}
  //                 <button
  //                   type="button"
  //                   onClick={togglePasswordVisibility}
  //                   className="absolute inset-y-0 right-0 flex items-center px-3 text-gray-500 hover:text-gray-700 focus:outline-none"
  //                   aria-label={
  //                     showPassword ? "Hide password" : "Show password"
  //                   }
  //                 >
  //                   {showPassword ? (
  //                     <EyeOff className="h-5 w-5" />
  //                   ) : (
  //                     <Eye className="h-5 w-5" />
  //                   )}
  //                 </button>
  //               </div>
  //               {errors.password && (
  //                 <p className="mt-1 text-xs text-red-600">
  //                   {errors.password.message}
  //                 </p> // Adjusted error style
  //               )}
  //             </div>
  //           )}
  //           {/* Submit Button */}
  //           {hideManualLogin == false && (
  //             <Button
  //               type="submit"
  //               disabled={mutation.isPending}
  //               className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-[#4abf6b] hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50"
  //             >
  //               {mutation.isPending ? (
  //                 <>
  //                   <svg
  //                     className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
  //                     xmlns="http://www.w3.org/2000/svg"
  //                     fill="none"
  //                     viewBox="0 0 24 24"
  //                   >
  //                     <circle
  //                       className="opacity-25"
  //                       cx="12"
  //                       cy="12"
  //                       r="10"
  //                       stroke="currentColor"
  //                       strokeWidth="4"
  //                     ></circle>
  //                     <path
  //                       className="opacity-75"
  //                       fill="currentColor"
  //                       d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
  //                     ></path>
  //                   </svg>
  //                   Logging in...
  //                 </>
  //               ) : (
  //                 "Login"
  //               )}
  //             </Button>
  //           )}
  //           {hideManualLogin == false && (
  //             <Button
  //               type="button"
  //               variant="outline"
  //               onClick={() => {
  //                 window.location.href = `${process.env.NEXT_PUBLIC_BACKEND_API_URL}/backend/api/v1/auth/sso/login`;
  //               }}
  //               className="w-full flex justify-center py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-[#003057] bg-white hover:bg-gray-100 mt-2"
  //             >
  //               Login with Okta
  //             </Button>
  //           )}
  //         </form>
  //       </div>
  //     </div>
  //   </div>
  // );

  return (
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
}
