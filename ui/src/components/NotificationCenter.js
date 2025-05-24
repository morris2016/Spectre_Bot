import React, { useEffect, useRef } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { useSnackbar } from 'notistack';
import { actions as alertsActions } from '../store/slices/alertsSlice';

/**
 * NotificationCenter displays alerts from Redux using snackbar toasts.
 * Alerts are removed from the store when dismissed.
 */
const NotificationCenter = () => {
  const alerts = useSelector((state) => state.alerts.list);
  const dispatch = useDispatch();
  const { enqueueSnackbar } = useSnackbar();
  const displayed = useRef([]);

  useEffect(() => {
    alerts.forEach((alert) => {
      if (displayed.current.includes(alert.id)) return;
      enqueueSnackbar(alert.message, {
        variant: alert.severity || 'info',
        onClose: (_, reason) => {
          if (reason === 'clickaway') return;
          dispatch(alertsActions.removeAlert(alert.id));
        },
        onExited: () => {
          dispatch(alertsActions.removeAlert(alert.id));
        },
      });
      displayed.current.push(alert.id);
    });
  }, [alerts, enqueueSnackbar, dispatch]);

  useEffect(() => {
    displayed.current = displayed.current.filter((id) =>
      alerts.some((alert) => alert.id === id)
    );
  }, [alerts]);

  return null;
};

export default NotificationCenter;
