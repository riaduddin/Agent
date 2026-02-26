import { useState, useEffect } from 'react';
import axiosInstance from '@/lib/axiosInstance';

interface GcsConfig {
  sourceRoot: string;
  bucketName: string;
}

export const useGcsConfig = () => {
  const [config, setConfig] = useState<GcsConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        setLoading(true);
        const response = await axiosInstance.get('/gcs/config');
        setConfig(response.data);
        setError(null);
      } catch (err) {
        setError(err as Error);
        console.error('Error fetching GCS config:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchConfig();
  }, []);

  return { config, loading, error };
};

export default useGcsConfig;
