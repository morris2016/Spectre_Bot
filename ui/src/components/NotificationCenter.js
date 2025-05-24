import React, { useEffect, useRef } from 'react';
import { useSnackbar } from 'notistack';
import { useDispatch, useSelector } from 'react-redux';
import { actions as alertsActions } from '../store/slices/alertsSlice';

/**
 * Subscribe to Redux alerts and display them via snackbars.
 */
const NotificationCenter = () => {
  const dispatch = useDispatch();
  const { enqueueSnackbar, closeSnackbar } = useSnackbar();
  const alerts = useSelector((state) => state.alerts.list);
  const shown = useRef(new Set());

  useEffect(() => {
    alerts.forEach((alert) => {
      if (shown.current.has(alert.id)) return;
      shown.current.add(alert.id);
      enqueueSnackbar(alert.message, {
        variant: alert.type || 'info',
        autoHideDuration: alert.timeout || 5000,
        onClose: () => dispatch(alertsActions.removeAlert(alert.id)),
      });
    });
  }, [alerts, enqueueSnackbar, dispatch]);

  useEffect(() => () => shown.current.clear(), []);

  return null;
};

export default NotificationCenter;
