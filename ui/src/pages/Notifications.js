import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import {
  Box,
  Typography,
  IconButton,

  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Button,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import { actions as alertsActions } from '../store/slices/alertsSlice';

/**
 * Notifications history page.

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

  const handleClear = () => dispatch(alertsActions.clearAlerts());
  const handleDismiss = (id) => dispatch(alertsActions.removeAlert(id));
  const handleRemove = (id) => dispatch(alertsActions.removeAlert(id));
  const handleClear = () => dispatch(alertsActions.clearAlerts());

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Typography variant="h4">Notifications</Typography>
      <Paper>
        <List dense>
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
                primaryTypographyProps={{ color: alert.type === 'error' ? 'error' : 'inherit' }}
              />
              <ListItemSecondaryAction>
                <IconButton edge="end" aria-label="delete" onClick={() => handleDismiss(alert.id)}>
                  <DeleteIcon />
              />
              <ListItemSecondaryAction>
                <IconButton edge="end" onClick={() => handleRemove(alert.id)}>
                  <CloseIcon />
                </IconButton>
              </ListItemSecondaryAction>
            </ListItem>
          ))}
          {alerts.length === 0 && (
            <ListItem>
              <ListItemText primary="No notifications" />
            </ListItem>
          )}
        </List>
      </Paper>
      {alerts.length > 0 && (
        <Button variant="contained" color="primary" onClick={handleClear} sx={{ alignSelf: 'flex-start' }}>
          Clear All
        </Button>
      )}
        </List>
      </Paper>
    </Box>
  );
};

export default Notifications;
