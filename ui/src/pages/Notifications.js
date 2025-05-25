import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Button,
  Paper,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import { actions as alertsActions } from '../store/slices/alertsSlice';

/**
 * Notifications page lists all alerts with options to dismiss them.
 */
const Notifications = () => {
  const alerts = useSelector((state) => state.alerts.list);
  const dispatch = useDispatch();

import { useSelector, useDispatch } from 'react-redux';
import {
  Box,
  Typography,
  Paper,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Button,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useDispatch, useSelector } from 'react-redux';
import { actions as alertsActions } from '../store/slices/alertsSlice';

/**
 * Notifications history page with dismiss and clear options.
 */
const Notifications = () => {
  const dispatch = useDispatch();
  const alerts = useSelector((state) => state.alerts.list);

  const handleDismiss = (id) => {
    dispatch(alertsActions.removeAlert(id));
  };

  const handleClear = () => {
    dispatch(alertsActions.clearAlerts());
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', mt: 2 }}>
      <Typography variant="h4" gutterBottom>
        Notifications
      </Typography>
      <Paper sx={{ p: 2 }}>
        {alerts.length === 0 ? (
          <Typography variant="body2">No notifications</Typography>
        ) : (
          <List>
            {alerts.map((alert) => (
              <ListItem key={alert.id} divider>
                <ListItemText
                  primary={alert.message}
                  secondary={
                    alert.timestamp
                      ? new Date(alert.timestamp).toLocaleString()
                      : null
                  }
                />
                <ListItemSecondaryAction>
                  <IconButton
                    edge="end"
                    aria-label="delete"
                    onClick={() => handleDismiss(alert.id)}
                  >
                    <DeleteIcon />
                  </IconButton>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
        )}
        {alerts.length > 0 && (
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 1 }}>
            <Button variant="outlined" onClick={handleClear} size="small">
              Clear All
            </Button>
          </Box>
        )}

  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Button,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { alertsActions } from '../store';

/**
 * Notifications page renders all alerts stored in Redux state and allows
 * users to dismiss individual items or clear the entire list.
 */
const Notifications = () => {
  const alerts = useSelector((state) => state.alerts.list);
  const dispatch = useDispatch();

  const handleRemove = (id) => dispatch(alertsActions.removeAlert(id));
  const handleClear = () => dispatch(alertsActions.clearAlerts());

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Typography variant="h4">Notifications</Typography>
      <Paper sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 1 }}>
          <Button onClick={handleClear} disabled={alerts.length === 0}>
            Clear All
          </Button>
        </Box>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Type</TableCell>
              <TableCell>Message</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {alerts.map((alert) => (
              <TableRow key={alert.id}>
                <TableCell>{alert.type}</TableCell>
                <TableCell>{alert.message}</TableCell>
                <TableCell align="right">
                  <IconButton size="small" onClick={() => handleDismiss(alert.id)}>
                    <CloseIcon fontSize="small" />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
            {alerts.length === 0 && (
              <TableRow>
                <TableCell colSpan={3} align="center">
                  No notifications
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
        <Box sx={{ textAlign: 'right', mb: 1 }}>
          <Button size="small" variant="outlined" onClick={handleClear}>
            Clear All
          </Button>
        </Box>
        <List dense>
          {alerts.length === 0 && (
            <Typography variant="body2" sx={{ p: 1 }}>
              No notifications
            </Typography>
          )}
          {alerts.map((alert) => (
            <ListItem key={alert.id} divider>
              <ListItemText
                primary={alert.message}
                secondary={alert.details}
              />
              <ListItemSecondaryAction>
                <IconButton edge="end" onClick={() => handleRemove(alert.id)}>
                  <CloseIcon />
                </IconButton>
              </ListItemSecondaryAction>
            </ListItem>
          ))}
        </List>
      </Paper>
    </Box>
  );
};

export default Notifications;
