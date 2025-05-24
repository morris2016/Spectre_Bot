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
 */
const Notifications = () => {
  const alerts = useSelector((state) => state.alerts.list);
  const dispatch = useDispatch();

  const handleClear = () => dispatch(alertsActions.clearAlerts());
  const handleDismiss = (id) => dispatch(alertsActions.removeAlert(id));

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Typography variant="h4">Notifications</Typography>
      <Paper>
        <List dense>
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
    </Box>
  );
};

export default Notifications;
