import { Suspense } from 'react'
import SsoCallback from '@/components/auth/SsoCallback'

export default function SSOCallbackPage() {
  return (
    <Suspense fallback={<p className="text-center mt-10">Loading...</p>}>
      <SsoCallback />
    </Suspense>
  )
}
