// src/app/layout.tsx (ROOT LAYOUT)
"use client"; // Provider needs client context

import { Geist, Geist_Mono, Open_Sans } from "next/font/google"; // Removed Montserrat
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'; // Import React Query
import "./globals.css"; // Import global styles here
import "./dhs_design_system.css"; // Import DHS design system
import { Toaster } from 'sonner'; // Import Toaster

import { AuthProvider } from '@/context/AuthContext'; // Import AuthProvider
import { usePathValidation } from "@/hooks/use-path-validation"; // Import path validation hook

// Define fonts globally
const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});
const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});
const openSans = Open_Sans({
  variable: "--font-open-sans",
  subsets: ["latin"],
});
const montserrat = Open_Sans({ // Alias for compatibility
  variable: "--unnamed-font-family-montserrat",
  subsets: ["latin"],
});

// Basic metadata (can be overridden by pages)
// export const metadata: Metadata = { // Cannot export metadata from client component root layout easily
//   title: "Georgia Document Digitization Platform",
//   description: "Document digitization and access platform.",
// };

// Create a QueryClient instance (ensure it's stable across renders)
// It's often recommended to create this outside the component or use useState
// to prevent recreation on every render. For simplicity here, we create it once.
const queryClient = new QueryClient();

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // Invoke the path validation hook to check routes globally
  usePathValidation();

  return (
    <html lang="en">
      <body className={`${openSans.variable} ${geistMono.variable} ${montserrat.variable} antialiased`}>
        {/* Wrap entire application with QueryClientProvider */}
        <QueryClientProvider client={queryClient}>
          {/* Wrap children with AuthProvider */}
          <AuthProvider>
            {children} {/* Child layouts (public or private group) will render here */}
            <Toaster richColors position="top-right" />
          </AuthProvider>
        </QueryClientProvider>
      </body>
    </html>
  );
}
