"use client";

import React, { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation'; // Import useRouter
import { useAuth } from '@/context/AuthContext';
import axiosInstance from '@/lib/axiosInstance';
import { Button } from '@/components/ui/button'; // Import Button
import { ArrowLeft } from 'lucide-react'; // Import ArrowLeft icon
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Terminal } from "lucide-react";
import { format } from 'date-fns';

interface SystemLog {
  log_id: string;
  session_id: string;
  message_id: string;
  user_query_text: string;
  timestamp: string; // ISO string
  step_name: string;
  status: string;
  step_details: Record<string, any>;
  error_message?: string;
}

interface PageParams {
  session_id?: string; // Make optional to handle initial undefined state
  message_id?: string; // Make optional to handle initial undefined state
}

const MessageSystemLogsPage = () => {
  const params = useParams();
  const router = useRouter(); // Initialize useRouter
  const session_id = params.session_id as string | undefined;
  const message_id = params.message_id as string | undefined;
  const { user } = useAuth(); // Removed authLoading, will rely on user object
  const [logs, setLogs] = useState<SystemLog[]>([]);
  const [isLoading, setIsLoading] = useState(true); // This will be the primary loading state
  const [error, setError] = useState<string | null>(null);
  const [userQuery, setUserQuery] = useState<string | null>(null);
  type MessageType = { textResponseTime?: string; referenceTime?: string } | null;
  const [message, setMessage] = useState<MessageType>(null);



  useEffect(() => {
    // Check if user is loaded (not null) and params are available
    if (user && session_id && message_id) {
      const fetchLogs = async () => {
        setIsLoading(true); // Keep this for the data fetching operation
        setError(null);
        try {
          const response = await axiosInstance.get(`/logs/sessions/${session_id}/messages/${message_id}/system_logs`);
          setLogs(response.data.logs || []);

          if (response.data.message) {
            setMessage(response.data.message);
          }
          if (response.data.logs && response.data.logs.length > 0) {
            setUserQuery(response.data.logs[0].user_query_text);
          }
        } catch (err: any) {
          console.error("Error fetching system logs:", err);
          setError(err.response?.data?.msg || "Failed to fetch system logs.");
        } finally {
          setIsLoading(false);
        }
      };
      fetchLogs();
    } else if (session_id && message_id && user === null) { // User is explicitly null (not logged in after auth check)
        setError("You must be logged in to view system logs.");
        setIsLoading(false); // Stop loading as we know user is not logged in
    } else if (!session_id || !message_id) { // Params not yet available
        setIsLoading(true); // Keep loading until params are available
    }
  }, [session_id, message_id, user]); // Depend on user, session_id, message_id

  // Combined loading state: wait for user and params, then for data fetching
  if (isLoading || !user || !session_id || !message_id && !error) {
    // If there's an error already (like not logged in), don't show loading
    if (error && !isLoading) { 
        // Fall through to render error
    } else {
        return (
          <div className="flex justify-center items-center h-screen">
            <p>Loading system logs...</p>
          </div>
        );
    }
  }

  return (
    <div className="container mx-auto p-4 md:p-8">
      <div className="flex items-center mb-6">
        <Button variant="outline" size="icon" onClick={() => router.back()} className="mr-4">
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <div className="flex-1">
          <h1 className="text-2xl font-bold">System Logs for Message</h1>
          <p className="text-sm text-muted-foreground">
            Detailed processing steps for a specific chat message.
          </p>
        </div>
      </div>
      
      <Card className="mb-6">
        <CardHeader>
          {/* <CardTitle>System Logs for Message</CardTitle> // Title moved above */}
          <CardDescription>
            Contextual information for these logs.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p><strong>Session ID:</strong> {session_id}</p>
          <p><strong>Response time: </strong> {message && message["textResponseTime"]} sec</p>
          <p><strong>Reference found: </strong> {message && message["referenceTime"]} sec</p>
          {userQuery && <p><strong>User Query:</strong> {userQuery}</p>}
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive" className="mb-4">
          <Terminal className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {!error && logs.length === 0 && !isLoading && (
        <Alert>
          <Terminal className="h-4 w-4" />
          <AlertTitle>No Logs Found</AlertTitle>
          <AlertDescription>
            No system logs were found for this message, or they are still being generated.
          </AlertDescription>
        </Alert>
      )}

      {!error && logs.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Log Entries</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[200px]">Timestamp</TableHead>
                  <TableHead>Step Name</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Details</TableHead>
                  <TableHead>Error</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {logs.map((log) => (
                  <TableRow key={log.log_id}>
                    <TableCell>
                      {format(new Date(log.timestamp), "yyyy-MM-dd HH:mm:ss.SSS")}
                    </TableCell>
                    <TableCell>{log.step_name}</TableCell>
                    <TableCell>
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                            log.status === 'SUCCESS' ? 'bg-green-100 text-green-800' :
                            log.status === 'ERROR' ? 'bg-red-100 text-red-800' :
                            log.status === 'INFO' ? 'bg-blue-100 text-blue-800' :
                            log.status === 'WARNING' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-gray-100 text-gray-800'
                        }`}
                        >
                        {log.status}
                        </span>
                    </TableCell>
                    <TableCell>
                      <pre className="text-xs bg-gray-50 p-2 rounded overflow-x-auto max-w-md">
                        {JSON.stringify(log.step_details, null, 2)}
                      </pre>
                    </TableCell>
                    <TableCell>
                      {log.error_message ? (
                        <pre className="text-xs text-red-600 bg-red-50 p-2 rounded overflow-x-auto max-w-md">
                          {log.error_message}
                        </pre>
                      ) : '-'}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default MessageSystemLogsPage;
