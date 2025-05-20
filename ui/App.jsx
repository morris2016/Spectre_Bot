

/**
 * QuantumSpectre Elite Trading System
 * Main Application Component
 * 
 * This component serves as the main layout container for the application,
 * handling routing, authentication state, and global UI elements.
 */

import React, { useEffect, useState, Suspense, useCallback } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { Box, CircularProgress, useMediaQuery } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { useSnackbar } from 'notistack';

// Layout components
import MainLayout from './layouts/MainLayout';
import MinimalLayout from './layouts/MinimalLayout';

// Authentication
import { useAuth } from './contexts/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';

// Websocket connection
import { useWebSocket } from './contexts/WebSocketContext';

// Voice advisor
import { useVoiceAdvisor } from './contexts/VoiceAdvisorContext';
import VoiceAdvisorControls from './components/VoiceAdvisor/VoiceAdvisorControls';

// Actions
import { setSystemStatus } from './store/slices/systemSlice';
import { setActiveExchange } from './store/slices/exchangeSlice';

// Lazy loaded components for code splitting and performance
const Dashboard = React.lazy(() => import('./pages/Dashboard'));
const BinanceTradingView = React.lazy(() => import('./pages/BinanceTradingView'));
const DerivTradingView = React.lazy(() => import('./pages/DerivTradingView'));
const StrategyBuilder = React.lazy(() => import('./pages/StrategyBuilder'));
const Backtester = React.lazy(() => import('./pages/Backtester'));
const Analytics = React.lazy(() => import('./pages/Analytics'));
const Settings = React.lazy(() => import('./pages/Settings'));
const Login = React.lazy(() => import('./pages/Login'));
const Register = React.lazy(() => import('./pages/Register'));
const ForgotPassword = React.lazy(() => import('./pages/ForgotPassword'));
const NotFound = React.lazy(() => import('./pages/NotFound'));
const SystemMonitor = React.lazy(() => import('./pages/SystemMonitor'));
const UserProfile = React.lazy(() => import('./pages/UserProfile'));
const TradingJournal = React.lazy(() => import('./pages/TradingJournal'));
const LoopholeDetector = React.lazy(() => import('./pages/LoopholeDetector'));
const AssetExplorer = React.lazy(() => import('./pages/AssetExplorer'));

// Loading fallback component
const LoadingFallback = () => (
  
    
  
);

const App = () => {
  const theme = useTheme();
  const location = useLocation();
  const dispatch = useDispatch();
  const { enqueueSnackbar } = useSnackbar();
  const { isAuthenticated, user, loading: authLoading } = useAuth();
  const { connect, lastMessage, connectionStatus } = useWebSocket();
  const { initialize: initializeVoiceAdvisor, status: voiceStatus } = useVoiceAdvisor();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  const systemStatus = useSelector(state => state.system.status);
  const activeExchange = useSelector(state => state.exchange.activeExchange);
  
  const [isAppReady, setIsAppReady] = useState(false);

  // Initialize websocket connection
  useEffect(() => {
    if (isAuthenticated && !authLoading) {
      connect();
    }
  }, [isAuthenticated, authLoading, connect]);

  // Process websocket messages
  useEffect(() => {
    if (lastMessage) {
      try {
        const data = JSON.parse(lastMessage.data);
        
        // Handle different message types
        switch (data.type) {
          case 'system_status':
            dispatch(setSystemStatus(data.payload));
            break;
          case 'trading_signal':
            processTradingSignal(data.payload);
            break;
          case 'voice_advisory':
            processVoiceAdvisory(data.payload);
            break;
          case 'notification':
            enqueueSnackbar(data.payload.message, { 
              variant: data.payload.severity || 'info',
              autoHideDuration: data.payload.duration || 5000
            });
            break;
          default:
            // Handle other message types or forward to specific components
            break;
        }
      } catch (err) {
        console.error('Error processing websocket message:', err);
      }
    }
  }, [lastMessage, dispatch, enqueueSnackbar]);

  // Initialize voice advisor
  useEffect(() => {
    if (isAuthenticated && !authLoading) {
      initializeVoiceAdvisor();
    }
  }, [isAuthenticated, authLoading, initializeVoiceAdvisor]);

  // Process trading signals
  const processTradingSignal = useCallback((signal) => {
    // Logic to process incoming trading signals
    console.log('Received trading signal:', signal);
    
    // Play alert sound for high confidence signals
    if (signal.confidence > 0.8) {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      oscillator.type = 'sine';
      oscillator.frequency.setValueAtTime(signal.direction === 'buy' ? 800 : 600, audioContext.currentTime);
      const gainNode = audioContext.createGain();
      gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);
      oscillator.start();
      oscillator.stop(audioContext.currentTime + 0.3);
    }
    
    // Notify user based on signal importance
    if (signal.importance === 'high') {
      enqueueSnackbar(`${signal.direction.toUpperCase()} signal for ${signal.asset} with ${(signal.confidence * 100).toFixed(1)}% confidence`, {
        variant: 'success',
        autoHideDuration: 8000,
        action: (key) => (
           navigateToSignal(signal)}>
            View
          
        )
      });
    }
  }, [enqueueSnackbar]);

  // Process voice advisory messages
  const processVoiceAdvisory = useCallback((advisory) => {
    // Voice advisor will handle this through the context
    console.log('Received voice advisory:', advisory);
  }, []);

  // Navigate to trading view with signal context
  const navigateToSignal = useCallback((signal) => {
    // Navigate to appropriate trading view and focus on the signal
    const exchange = signal.exchange.toLowerCase();
    if (exchange === 'binance') {
      dispatch(setActiveExchange('binance'));
      window.location.href = `/trading/binance?asset=${signal.asset}&signal=${signal.id}`;
    } else if (exchange === 'deriv') {
      dispatch(setActiveExchange('deriv'));
      window.location.href = `/trading/deriv?asset=${signal.asset}&signal=${signal.id}`;
    }
  }, [dispatch]);
  
  // Set app as ready after initial loading
  useEffect(() => {
    if (!authLoading) {
      setIsAppReady(true);
    }
  }, [authLoading]);

  // Handle app not ready state
  if (!isAppReady) {
    return ;
  }

  return (
    <>
      }>
        
          {/* Public routes */}
          }>
             : } />
             : } />
             : } />
          

          {/* Protected routes */}
          }>
            } />
            } />
            } />
            } />
            } />
            } />
            } />
            } />
            } />
            } />
            } />
            } />
            } />
          

          {/* 404 route */}
          } />
        
      

      {/* Global Voice Advisor Controls - Only show when authenticated */}
      {isAuthenticated && }
    
  );
};

export default App;

