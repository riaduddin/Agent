export interface Transfer {
  id: string;
  fileName: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'partial';
  progress: number;
  source: string;
  destination: string;
  timestamp: string;
  operation?: 'move' | 'copy'; // Add operation field
  transfer_type?: string; // "Folder" tag for folder transfers, undefined for regular file transfers
}
