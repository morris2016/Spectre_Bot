import React, { useEffect, useState } from 'react';
import { Box, Typography, Paper, Table, TableBody, TableCell, TableHead, TableRow } from '@mui/material';
import TradingView from '../components/TradingView/TradingView';
import OrderPanel from '../components/OrderPanel/OrderPanel';
import api from '../api';

/**
 * Dashboard page showing market overview and quick trading actions.
 */
const Dashboard = () => {
  const [marketData, setMarketData] = useState(null);
  const [openOrders, setOpenOrders] = useState([]);
  const [positions, setPositions] = useState([]);
  const [analytics, setAnalytics] = useState(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [tickerRes, ordersRes, positionsRes, analyticsRes] = await Promise.all([
          api.market.getTicker({ symbol: 'BTCUSDT' }),
          api.trading.getOpenOrders({}),
          api.trading.getPositions({}),
          api.portfolio.getPerformance({ period: '1d' }),
        ]);
        setMarketData(tickerRes.data);
        setOpenOrders(ordersRes.data || []);
        setPositions(positionsRes.data || []);
        setAnalytics(analyticsRes.data);
      } catch (error) {
        console.error('Failed to load dashboard data', error);
      }
    };

    loadData();
  }, []);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Typography variant="h4">Dashboard</Typography>
      <Box sx={{ height: 400 }}>
        <TradingView />
      </Box>
      <OrderPanel />

      {analytics && (
        <Paper sx={{ p: 2, mb: 2 }}>
          <Typography variant="h6">Daily Performance</Typography>
          <Typography variant="body2">PNL: {analytics.pnl}</Typography>
          <Typography variant="body2">Return: {analytics.return}%</Typography>
        </Paper>
      )}

      <Paper sx={{ p: 2, mb: 2 }}>
        <Typography variant="h6">Open Positions</Typography>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Symbol</TableCell>
              <TableCell>Quantity</TableCell>
              <TableCell>Entry</TableCell>
              <TableCell>PnL</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {positions.map((p) => (
              <TableRow key={p.id}>
                <TableCell>{p.symbol}</TableCell>
                <TableCell>{p.quantity}</TableCell>
                <TableCell>{p.entryPrice}</TableCell>
                <TableCell>{p.unrealizedPnl}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>

      <Paper sx={{ p: 2 }}>
        <Typography variant="h6">Open Orders</Typography>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>ID</TableCell>
              <TableCell>Symbol</TableCell>
              <TableCell>Side</TableCell>
              <TableCell>Price</TableCell>
              <TableCell>Quantity</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {openOrders.map((o) => (
              <TableRow key={o.id}>
                <TableCell>{o.id}</TableCell>
                <TableCell>{o.symbol}</TableCell>
                <TableCell>{o.side}</TableCell>
                <TableCell>{o.price}</TableCell>
                <TableCell>{o.quantity}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>

      {marketData && (
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6">Latest Price</Typography>
          <Typography variant="body2">{marketData.lastPrice}</Typography>
        </Paper>
      )}
    </Box>
  );
};

export default Dashboard;
