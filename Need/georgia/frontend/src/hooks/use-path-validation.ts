"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";

export function usePathValidation() {
    const pathname = usePathname();
    // const router = useRouter(); // Removed router usage
    const [isRedirecting, setIsRedirecting] = useState(true); // Default to true to prevent flash

    useEffect(() => {
        // Logic:
        // If the ingress/basePath configuration strips the base path before hitting Next.js, 
        // pathname might be just "/" or "/login".
        // If Next.js sees it safely with base path, it might still report "/" relative to the app.

        // We want to force redirect if we are at root or /login
        // The target is strictly configured to be /admin/login (which will be relative to configured HOST_URL)

        // Delay check by 1 second to ensure page/network settles
        const timer = setTimeout(() => {
            const path_name = `${process.env.NEXT_PUBLIC_BASE_PATH}${pathname}`;
            console.log(path_name, ": Checking path after 3s delay");

            if (path_name === `${process.env.NEXT_PUBLIC_BASE_PATH}/` || path_name === `${process.env.NEXT_PUBLIC_BASE_PATH}/login`) {
                // Force hard redirect using window.location.href to the configured HOST URL
                const hostUrl = process.env.NEXT_PUBLIC_HOST_URL || "";

                if (hostUrl) {
                    console.log("==== > Redirecting to Admin Login...");
                    window.location.href = `${hostUrl}/admin/login`;
                } else {
                    // Fallback logic if needed (currently commented out in original, keeping it out)
                    console.warn("HOST_URL not set, skipping redirect");
                }
            } else {
                // It's a genuine 404 or other page
                setIsRedirecting(false);
            }
        }, 3000);

        // Cleanup timer
        return () => clearTimeout(timer);
    }, [pathname]);

    return { isRedirecting };
}
