// src/app/register/layout.tsx
import PublicLayout from '@/components/layouts/PublicLayout';
import React from 'react';

export default function RegisterLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Apply the public layout to the register page
  return <PublicLayout>{children}</PublicLayout>;
}
