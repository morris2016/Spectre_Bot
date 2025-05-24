import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import {
  Box,
  Typography,
  Paper,
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
