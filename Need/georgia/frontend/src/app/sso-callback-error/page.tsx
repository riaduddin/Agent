'use client';

import { useRouter, useSearchParams } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { AlertTriangle } from 'lucide-react';
import { Suspense } from 'react';

function SsoCallbackErrorContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const errorMessage = searchParams.get('message') || 'An unknown error occurred during SSO login.';

  const handleReturnToLogin = () => {
    router.push('/login');
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <Card className="w-full max-w-md mx-4">
        <CardHeader>
          <CardTitle className="flex items-center justify-center text-center text-2xl text-red-600">
            <AlertTriangle className="mr-2 h-8 w-8" />
            SSO Login Failed
          </CardTitle>
        </CardHeader>
        <CardContent className="text-center">
          <p className="text-gray-700 mb-6">
            {errorMessage}
          </p>
          <Button onClick={handleReturnToLogin} className="w-full">
            Return to Login
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

export default function SsoCallbackErrorPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <SsoCallbackErrorContent />
    </Suspense>
  );
}
