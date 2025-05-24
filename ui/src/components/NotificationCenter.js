import React, { useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { useSnackbar } from 'notistack';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import { actions as alertsActions } from '../store/slices/alertsSlice';

/**
 * NotificationCenter renders Redux alerts as dismissible snackbars.
 */
const NotificationCenter = () => {
  const alerts = useSelector((state) => state.alerts.list);
  const dispatch = useDispatch();
  const { enqueueSnackbar, closeSnackbar } = useSnackbar();

  useEffect(() => {
    alerts.forEach((alert) => {
      enqueueSnackbar(alert.message, {
        variant: alert.type || 'default',
        autoHideDuration: alert.timeout || 5000,
        action: (key) => (
          <IconButton
            size="small"
            aria-label="close"
            color="inherit"
            onClick={() => {
              closeSnackbar(key);
              dispatch(alertsActions.removeAlert(alert.id));
            }}
          >
            <CloseIcon fontSize="small" />
          </IconButton>
        ),
        onExited: () => dispatch(alertsActions.removeAlert(alert.id)),
      });
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [alerts]);

  return null;
};

export default NotificationCenter;
