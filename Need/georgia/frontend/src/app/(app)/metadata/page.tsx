// src/app/(app)/metadata/page.tsx - Moved to private route group
"use client";

import withAuth from "@/components/auth/withAuth"; // Import the HOC
import React from 'react';

function MetadataPage() {
  return (
    <div className="p-4">
      <h1 className="text-2xl font-semibold mb-4 text-dark-gray">
        Document Metadata
      </h1>
      {/* TODO: Implement table component to display Firestore metadata */}
      <div className="bg-white p-6 rounded shadow text-dark-gray">
        <p>Metadata table will be displayed here.</p>
        {/* Example structure - replace with actual table */}
        <table className="w-full mt-4 text-left border-collapse">
          <thead>
            <tr className="border-b border-medium-gray">
              <th className="py-2 px-4">Filename</th>
              <th className="py-2 px-4">Size</th>
              <th className="py-2 px-4">Upload Date</th>
              <th className="py-2 px-4">Status</th>
            </tr>
          </thead>
          <tbody>
            {/* Add table rows dynamically */}
            <tr className="border-b border-light-gray-2">
              <td className="py-2 px-4">example_doc_1.pdf</td>
              <td className="py-2 px-4">1.2 MB</td>
              <td className="py-2 px-4">2025-03-30</td>
              <td className="py-2 px-4">Processed</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default withAuth(MetadataPage); // Wrap component
