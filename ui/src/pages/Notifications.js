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
      </Paper>
    </Box>
  );
};

export default Notifications;
