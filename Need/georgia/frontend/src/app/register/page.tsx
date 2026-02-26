"use client"; // Needed for hooks

// import React from 'react';
// import { useForm, SubmitHandler } from 'react-hook-form';
// import { useMutation } from '@tanstack/react-query';
// import { isAxiosError } from 'axios'; // Import axios and isAxiosError
// import axiosInstance from '@/lib/axiosInstance'; // Import configured Axios
// import { useRouter } from 'next/navigation'; // For redirection
// import Link from 'next/link'; // For link to login

// // Define the type for our form inputs
// type RegisterFormInputs = {
//   name?: string; // Optional name field
//   email: string;
//   password: string;
//   confirmPassword: string; // For password confirmation
// };

// // Define the expected API response structure (adjust as needed)
// type RegisterApiResponse = {
//   msg: string;
//   user_id: string; // Or whatever the backend returns on success
// };

// // Define the mutation function
// const registerUser = async (data: Omit<RegisterFormInputs, 'confirmPassword'>): Promise<RegisterApiResponse> => {
//   // Don't send confirmPassword to the backend
//   const { name, email, password } = data;
//   const response = await axiosInstance.post<RegisterApiResponse>('/auth/register', { name, email, password });
//   return response.data;
// };

// export default function RegisterPage() {
//   const router = useRouter();
//   const {
//     register,
//     handleSubmit,
//     watch, // To watch confirmPassword field
//     formState: { errors },
//     setError, // Function to set errors manually
//   } = useForm<RegisterFormInputs>();

//   // Watch the password field to compare with confirmPassword
//   const password = watch('password');

//   const mutation = useMutation<RegisterApiResponse, Error, Omit<RegisterFormInputs, 'confirmPassword'>>({
//     mutationFn: registerUser,
//     onSuccess: (data) => {
//       // Optionally redirect to login or show success message
//       alert('Registration successful! Please log in.'); // Simple alert for now
//       router.push('/login'); // Redirect to login page
//     },
//     onError: (error: unknown) => { // Use unknown type
//       console.error('Registration error:', error);
//       let errorMessage = 'Registration failed. Please try again.';
//       // Use type guard for AxiosError
//       if (isAxiosError(error) && error.response?.data?.msg) {
//           errorMessage = error.response.data.msg;
//       } else if (error instanceof Error) {
//           errorMessage = error.message;
//       }
//       setError('root.serverError', {
//         type: 'manual',
//         message: errorMessage,
//       });
//     },
//   });

//   const onSubmit: SubmitHandler<RegisterFormInputs> = (data) => {
//      // We already validated passwords match in the form validation rules
//      // Prepare data for API call (exclude confirmPassword)
//      const apiData = { name: data.name, email: data.email, password: data.password };
//      mutation.mutate(apiData);
//   };

//   return (
//     <div className="flex items-center justify-center min-h-screen bg-ultra-light-gray">
//       <div className="w-full max-w-md p-8 space-y-6 bg-white rounded shadow-md">
//         <h2 className="text-2xl font-bold text-center text-dark-gray">Register</h2>
//         <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
//           {errors.root?.serverError && (
//             <p className="text-sm text-red-600 text-center">{errors.root.serverError.message}</p>
//           )}
//           <div>
//             <label htmlFor="name" className="block text-sm font-medium text-medium-gray">
//               Name (Optional)
//             </label>
//             <input
//               id="name"
//               type="text"
//               {...register('name')}
//               className="mt-1 block w-full px-3 py-2 border border-light-gray-2 rounded-md shadow-sm focus:outline-none focus:ring-1 focus:ring-accent-blue focus:border-accent-blue sm:text-sm"
//               placeholder="John Doe"
//             />
//             {/* No error display needed for optional field */}
//           </div>
//           <div>
//             <label htmlFor="email" className="block text-sm font-medium text-medium-gray">
//               Email address
//             </label>
//             <input
//               id="email"
//               type="email"
//               {...register('email', {
//                 required: 'Email is required',
//                 pattern: {
//                   value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
//                   message: 'Invalid email address',
//                 },
//               })}
//               className={`mt-1 block w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-1 sm:text-sm ${
//                 errors.email ? 'border-red-500 focus:ring-red-500 focus:border-red-500' : 'border-light-gray-2 focus:ring-accent-blue focus:border-accent-blue'
//               }`}
//               placeholder="you@example.com"
//             />
//             {errors.email && (
//               <p className="mt-1 text-sm text-red-600">{errors.email.message}</p>
//             )}
//           </div>
//           <div>
//             <label htmlFor="password"className="block text-sm font-medium text-medium-gray">
//               Password
//             </label>
//             <input
//               id="password"
//               type="password"
//               {...register('password', {
//                  required: 'Password is required',
//                  minLength: { value: 8, message: 'Password must be at least 8 characters' }
//               })}
//               className={`mt-1 block w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-1 sm:text-sm ${
//                 errors.password ? 'border-red-500 focus:ring-red-500 focus:border-red-500' : 'border-light-gray-2 focus:ring-accent-blue focus:border-accent-blue'
//               }`}
//               placeholder="********"
//             />
//             {errors.password && (
//               <p className="mt-1 text-sm text-red-600">{errors.password.message}</p>
//             )}
//           </div>
//            <div>
//             <label htmlFor="confirmPassword"className="block text-sm font-medium text-medium-gray">
//               Confirm Password
//             </label>
//             <input
//               id="confirmPassword"
//               type="password"
//               {...register('confirmPassword', {
//                  required: 'Please confirm your password',
//                  validate: value => value === password || 'Passwords do not match'
//               })}
//               className={`mt-1 block w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-1 sm:text-sm ${
//                 errors.confirmPassword ? 'border-red-500 focus:ring-red-500 focus:border-red-500' : 'border-light-gray-2 focus:ring-accent-blue focus:border-accent-blue'
//               }`}
//               placeholder="********"
//             />
//             {errors.confirmPassword && (
//               <p className="mt-1 text-sm text-red-600">{errors.confirmPassword.message}</p>
//             )}
//           </div>
//           <div>
//             <button
//               type="submit"
//               disabled={mutation.isPending}
//               className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-primary-bg bg-accent-blue hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-accent-blue disabled:opacity-50"
//             >
//               {mutation.isPending ? 'Registering...' : 'Register'}
//             </button>
//           </div>
//         </form>
//         <p className="text-sm text-center text-medium-gray">
//             Already have an account?{' '}
//             <Link href="/login" className="font-medium text-accent-blue hover:underline">
//               Login here
//             </Link>
//           </p>
//       </div>
//     </div>
//   );
// }
export default function EmptyPage() {
  return <p>Not Allowed</p>;
}