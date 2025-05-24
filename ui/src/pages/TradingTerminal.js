import React, { useEffect, useState } from 'react';
import { Box, Typography, Paper, Table, TableBody, TableCell, TableHead, TableRow } from '@mui/material';
import TradingView from '../components/TradingView/TradingView';
import OrderPanel from '../components/OrderPanel/OrderPanel';
import api from '../api';

/**
 * Full trading terminal with chart, positions and order management.
 */
const TradingTerminal = () => {
  const [openOrders, setOpenOrders] = useState([]);
  const [positions, setPositions] = useState([]);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [ordersRes, positionsRes] = await Promise.all([
          api.trading.getOpenOrders({}),
          api.trading.getPositions({}),
        ]);
        setOpenOrders(ordersRes.data || []);
        setPositions(positionsRes.data || []);
      } catch (error) {
        console.error('Failed to load trading terminal data', error);
      }
    };

    loadData();
  }, []);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Typography variant="h4">Trading Terminal</Typography>
      <Box sx={{ height: 500 }}>
        <TradingView />
      </Box>
      <OrderPanel />

      <Paper sx={{ p: 2 }}>
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
    </Box>
  );
};

export default TradingTerminal;
