/**
 * QuantumSpectre Elite Trading System
 * Main App Component
 *
 * This component serves as the root of the application UI.
 * It handles the main layout, routing, and application lifecycle.
 */

import React, { useEffect, useState, Suspense } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useSelector, useDispatch } from 'react-redux';
import { useTheme, styled } from '@mui/material/styles';
import { Box, CircularProgress, Backdrop, useMediaQuery } from '@mui/material';
import { useTranslation } from 'react-i18next';
import { useSnackbar } from 'notistack';

// Core components
import Navigation from './components/Navigation';
import Sidebar from './components/Sidebar';
import Footer from './components/Footer';
import LoadingScreen from './components/common/LoadingScreen';
import SystemStatusBar from './components/SystemStatusBar';
import VoiceAdvisorPanel from './components/VoiceAdvisorPanel';
import NotificationCenter from './components/NotificationCenter';
import PlatformSwitcher from './components/PlatformSwitcher';
import KeyboardShortcutsDialog from './components/KeyboardShortcutsDialog';
import ErrorBoundary from './components/common/ErrorBoundary';

// Auth-related components
import Login from './pages/auth/Login';
import Register from './pages/auth/Register';
import ResetPassword from './pages/auth/ResetPassword';
import VerifyEmail from './pages/auth/VerifyEmail';

// Main application pages (lazy-loaded)
const Dashboard = React.lazy(() => import('./pages/Dashboard'));
const TradingTerminal = React.lazy(() => import('./pages/TradingTerminal'));
const Portfolio = React.lazy(() => import('./pages/Portfolio'));
const StrategyBuilder = React.lazy(() => import('./pages/StrategyBuilder'));
const Backtesting = React.lazy(() => import('./pages/Backtesting'));
const Settings = React.lazy(() => import('./pages/Settings'));
const Analytics = React.lazy(() => import('./pages/Analytics'));
const AccountDetails = React.lazy(() => import('./pages/AccountDetails'));
const Notifications = React.lazy(() => import('./pages/Notifications'));
const MarketAnalysis = React.lazy(() => import('./pages/MarketAnalysis'));
const MlModelTraining = React.lazy(() => import('./pages/MlModelTraining'));
const BrainPerformance = React.lazy(() => import('./pages/BrainPerformance'));
const SystemMonitor = React.lazy(() => import('./pages/SystemMonitor'));
const PatternLibrary = React.lazy(() => import('./pages/PatternLibrary'));
const NewsAnalysis = React.lazy(() => import('./pages/NewsAnalysis'));

// Auth actions
import { checkAuthStatus } from './store/slices/authSlice';

// System actions
import { initializeSystem, checkSystemHealth } from './store/slices/systemSlice';
import { initializePreferences } from './store/slices/preferencesSlice';

// Hooks
import { useWorkspace } from './hooks/useWorkspace';
import { useWebSocket } from './hooks/useWebSocket';
import { useVoiceAdvisor } from './hooks/useVoiceAdvisor';
import { useSystemMonitor } from './hooks/useSystemMonitor';

// Utilities
import { PLATFORM_TYPES } from './constants';
import { detectHardwareCapabilities } from './utils/systemUtils';
import { setupKeyboardShortcuts } from './utils/keyboardShortcuts';

// Styled components
const AppContainer = styled(Box)(({ theme, sidebarOpen }) => ({
  display: 'flex',
  flexDirection: 'column',
  minHeight: '100vh',
  backgroundColor: theme.palette.background.default,
  color: theme.palette.text.primary,
  transition: theme.transitions.create(['margin'], {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
  ...(sidebarOpen && {
    [theme.breakpoints.up('md')]: {
      marginLeft: 260,
      transition: theme.transitions.create(['margin'], {
        easing: theme.transitions.easing.easeOut,
        duration: theme.transitions.duration.enteringScreen,
      }),
    },
  }),
}));

const MainContent = styled(Box)(({ theme }) => ({
  flexGrow: 1,
  padding: theme.spacing(3),
  display: 'flex',
  flexDirection: 'column',
  overflow: 'auto',
  minHeight: 'calc(100vh - 64px)', // Full height minus app bar
}));

const App = () => {
  const theme = useTheme();
  const dispatch = useDispatch();
  const { t } = useTranslation();
  const { enqueueSnackbar } = useSnackbar();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const { isAuthenticated, loading: authLoading } = useSelector((state) => state.auth || {});
  const { initialized, loading: systemLoading, error: systemError } = useSelector((state) => state.system || {});
  const { currentPlatform } = useSelector((state) => state.trading || {});

  const [sidebarOpen, setSidebarOpen] = useState(!isMobile);
  const [shortcutsDialogOpen, setShortcutsDialogOpen] = useState(false);

  const { saveWorkspace, loadWorkspace } = useWorkspace();
  const { connect, disconnect } = useWebSocket();
  const { initializeVoiceAdvisor } = useVoiceAdvisor();
  const { startMonitoring } = useSystemMonitor();

  useEffect(() => {
    dispatch(checkAuthStatus());
    const cleanupShortcuts = setupKeyboardShortcuts({
      showShortcuts: () => setShortcutsDialogOpen(true),
      toggleSidebar: () => setSidebarOpen((prev) => !prev),
    });
    detectHardwareCapabilities();
    return () => {
      cleanupShortcuts();
    };
  }, [dispatch]);

  useEffect(() => {
    if (isAuthenticated && !initialized && !systemLoading) {
      dispatch(initializeSystem());
      dispatch(initializePreferences());
      loadWorkspace();
      connect();
      initializeVoiceAdvisor();
      startMonitoring();
      const healthCheckInterval = setInterval(() => {
        dispatch(checkSystemHealth());
      }, 30000);
      return () => {
        clearInterval(healthCheckInterval);
        disconnect();
        saveWorkspace();
      };
    }
  }, [
    isAuthenticated,
    initialized,
    systemLoading,
    dispatch,
    connect,
    disconnect,
    saveWorkspace,
    loadWorkspace,
    initializeVoiceAdvisor,
    startMonitoring,
  ]);

  useEffect(() => {
    if (systemError) {
      enqueueSnackbar(t('errors.system_error', { message: systemError }), { variant: 'error' });
    }
  }, [systemError, enqueueSnackbar, t]);

  useEffect(() => {
    if (currentPlatform === PLATFORM_TYPES.BINANCE) {
      document.title = 'QuantumSpectre Elite - Binance';
    } else if (currentPlatform === PLATFORM_TYPES.DERIV) {
      document.title = 'QuantumSpectre Elite - Deriv';
    } else {
      document.title = 'QuantumSpectre Elite Trading System';
    }
  }, [currentPlatform]);

  const handleToggleSidebar = () => {
    setSidebarOpen((prev) => !prev);
  };

  if (authLoading || (isAuthenticated && systemLoading)) {
    return (
      <Backdrop open>
        <CircularProgress color="inherit" />
      </Backdrop>
    );
  }

  if (!isAuthenticated) {
    return (
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/reset-password" element={<ResetPassword />} />
        <Route path="/verify-email" element={<VerifyEmail />} />
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    );
  }

  return (
    <ErrorBoundary>
      <AppContainer sidebarOpen={sidebarOpen}>
        <Sidebar open={sidebarOpen} onClose={handleToggleSidebar} />
        <Navigation onToggleSidebar={handleToggleSidebar} />
        <PlatformSwitcher />
        <SystemStatusBar />
        <MainContent>
          <Suspense fallback={<LoadingScreen />}>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/terminal" element={<TradingTerminal />} />
              <Route path="/portfolio" element={<Portfolio />} />
              <Route path="/strategy-builder" element={<StrategyBuilder />} />
              <Route path="/backtesting" element={<Backtesting />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/account" element={<AccountDetails />} />
              <Route path="/notifications" element={<Notifications />} />
              <Route path="/market-analysis" element={<MarketAnalysis />} />
              <Route path="/ml-training" element={<MlModelTraining />} />
              <Route path="/brain-performance" element={<BrainPerformance />} />
              <Route path="/system-monitor" element={<SystemMonitor />} />
              <Route path="/pattern-library" element={<PatternLibrary />} />
              <Route path="/news-analysis" element={<NewsAnalysis />} />
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </Suspense>
          <Footer />
        </MainContent>
        <NotificationCenter />
        <VoiceAdvisorPanel />
        <KeyboardShortcutsDialog
          open={shortcutsDialogOpen}
          onClose={() => setShortcutsDialogOpen(false)}
        />
      </AppContainer>
    </ErrorBoundary>
  );
};

export default App;
