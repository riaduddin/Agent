// src/app/login/layout.tsx
import PublicLayout from '@/components/layouts/PublicLayout';
import React from 'react';

export default function LoginLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Apply the public layout to the login page
  return <PublicLayout>{children}</PublicLayout>;
}
