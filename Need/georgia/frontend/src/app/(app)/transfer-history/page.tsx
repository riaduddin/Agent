"use client";

import withAuth from "@/components/auth/withAuth";
import TransferHistory from "@/components/file-management/TransferHistory";

function TransferHistoryPage() {
    return (
        <div className="flex-1 bg-[#FFFFFF] p-8 overflow-y-auto">
            <div className="max-w-7xl mx-auto space-y-8">
                <TransferHistory />
            </div>
        </div>
    );
}

export default withAuth(TransferHistoryPage);
