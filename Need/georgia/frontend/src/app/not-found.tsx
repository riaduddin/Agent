"use client";

import Image from "next/image";
import { usePathValidation } from "@/hooks/use-path-validation";

export default function NotFound() {
    const { isRedirecting } = usePathValidation();

    const handleReturnToLogin = () => {
        const hostUrl = process.env.NEXT_PUBLIC_HOST_URL || "";

        if (hostUrl) {
            console.log("==== > Redirecting to Admin Login...");
            window.location.href = `${hostUrl}/admin/login`;
        } else {
            // Fallback
            const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "/r14";
            // Ensure basePath doesn't end with slash to avoid double slashes if any
            const safeBasePath = basePath.endsWith('/') ? basePath.slice(0, -1) : basePath;
            window.location.href = `${window.location.origin}${safeBasePath}/admin/login`;
        }
    };

    if (isRedirecting) {
        return (
            <div className="flex min-h-screen flex-col items-center justify-center bg-[#1F2A38]">
                <div className="flex flex-col items-center gap-6 text-center animate-pulse">
                    <div className="relative w-48 h-48 mb-4">
                        <Image
                            src={`${process.env.NEXT_PUBLIC_BASE_PATH}/DHS_White_vert.png`}
                            alt="Georgia DHS Logo"
                            fill
                            className="object-contain"
                            priority
                        />
                    </div>
                    <p className="text-[#E6E9EA] text-lg font-medium">Checking authentication...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="flex min-h-screen flex-col items-center justify-center bg-[#1F2A38] text-[#E6E9EA]">
            <div className="container flex max-w-[64rem] flex-col items-center gap-8 text-center p-4">

                {/* Logo Section */}
                <div className="relative w-64 h-64">
                    <Image
                        src={`${process.env.NEXT_PUBLIC_BASE_PATH}/DHS_White_vert.png`}
                        alt="Georgia DHS Logo"
                        fill
                        className="object-contain"
                        priority
                    />
                </div>

                <div className="space-y-2">
                    <h1 className="text-4xl font-bold tracking-tight">Page Not Found</h1>
                    <p className="text-[#B6BDC9] max-w-[42rem] mx-auto text-lg">
                        The page you are looking for does not exist or you do not have permission to view it.
                    </p>
                </div>

                <div className="mt-8">
                    <button
                        onClick={handleReturnToLogin}
                        className="inline-flex h-11 items-center justify-center rounded-md bg-white/10 hover:bg-white/20 border border-white/20 px-8 text-sm font-medium text-white shadow-lg transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50"
                    >
                        Return to Login
                    </button>
                </div>
            </div>
        </div>
    );
}
