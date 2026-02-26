// src/components/layouts/PublicLayout.tsx
"use client"; // If using client-side logic later

import React from 'react';
import "../../app/globals.css"; // Corrected path to global styles

// Import fonts if needed, or rely on RootLayout's import if nested (though usually not for public)
// import { Geist, Geist_Mono } from "next/font/google";
// const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
// const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export default function PublicLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    // Simple layout, maybe just applying background and centering
    // Using geistSans/Mono variables assumes they are globally available from RootLayout,
    // otherwise, import and apply them here if this layout is used independently.
     <div className={`min-h-screen flex items-center justify-center bg-ultra-light-gray`}>
        {children}
     </div>
  );
}
