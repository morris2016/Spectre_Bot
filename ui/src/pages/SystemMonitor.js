import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { api, socketManager } from '../api';

const SystemMonitor = () => {
  const [metrics, setMetrics] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadMetrics = async () => {
    setLoading(true);
    setError(null);
    try {
      const { data } = await api.system.getMetrics({ limit: 30 });
      setMetrics(data);
    } catch (err) {
      setError('Failed to load metrics');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadMetrics();
  }, []);

  useEffect(() => {
    const unsub = socketManager.subscribe('system:metrics', (data) => {
      setMetrics((prev) => [...prev.slice(-29), data]);
    });
    return () => unsub();
  }, []);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Typography variant="h4">System Monitor</Typography>
      {loading && <CircularProgress sx={{ alignSelf: 'center', mt: 2 }} />}
      {error && (
        <Typography color="error" variant="body2" sx={{ mb: 2 }}>
          {error}
        </Typography>
      )}
      <Paper sx={{ p: 2, height: 300 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={metrics} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <XAxis
              dataKey="timestamp"
              tickFormatter={(v) => new Date(v).toLocaleTimeString()}
            />
            <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
            <Tooltip labelFormatter={(v) => new Date(v).toLocaleString()} />
            <Line type="monotone" dataKey="cpu" stroke="#8884d8" dot={false} />
            <Line type="monotone" dataKey="memory" stroke="#82ca9d" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </Paper>
    </Box>
  );
};

export default SystemMonitor;
