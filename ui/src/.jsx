

/**
 * QuantumSpectre Elite Trading System
 * Frontend Main Entry Point
 * 
 * This file serves as the entry point for the React frontend application.
 * It sets up the React application with all necessary providers and global configuration.
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { Provider } from 'react-redux';
import { PersistGate } from 'redux-persist/integration/react';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { SnackbarProvider } from 'notistack';
import { LocalizationProvider } from '@mui/x-date-pickers';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';

import App from './App';
import { store, persistor } from './store';
import { darkTheme, lightTheme } from './theme';
import { AuthProvider } from './contexts/AuthContext';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { ThemeModeProvider, useThemeMode } from './contexts/ThemeModeContext';
import { VoiceAdvisorProvider } from './contexts/VoiceAdvisorContext';
import { ErrorBoundary } from './components/ErrorBoundary';
import LoadingScreen from './components/LoadingScreen';

// Import global styles
import './index.css';

// Setup performance monitoring
import { setupPerformanceMonitoring } from './utils/performance';
setupPerformanceMonitoring();

// Implement theme selection wrapper
const ThemedApp = () => {
  const { themeMode } = useThemeMode();
  const theme = themeMode === 'dark' ? darkTheme : lightTheme;

  return (
    
      
      
    
  );
};

// Setup React root with all providers
ReactDOM.createRoot(document.getElementById('root')).render(
  
    
      
        } persistor={persistor}>
          
            
              
                
                  
                    
                      
                        
                      
                    
                  
                
              
            
          
        
      
    
  
);

// Register service worker for offline capability and PWA support
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/service-worker.js')
      .then(registration => {
        console.log('SW registered: ', registration);
      })
      .catch(registrationError => {
        console.log('SW registration failed: ', registrationError);
      });
  });
}

// Setup GPU acceleration detection and optimization
import { checkGPUCapabilities, optimizeForGPU } from './utils/hardware';
checkGPUCapabilities().then(capabilities => {
  if (capabilities.gpuAcceleration) {
    optimizeForGPU(capabilities);
    console.log('GPU acceleration enabled:', capabilities.gpuInfo);
  } else {
    console.log('GPU acceleration not available, using fallback rendering');
  }
});

// Error tracking and reporting
window.addEventListener('error', (event) => {
  // Log to monitoring service
  console.error('Global error:', event.error);
  
  // Send to backend for tracking
  fetch('/api/v1/monitoring/error', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message: event.message,
      source: event.filename,
      line: event.lineno,
      column: event.colno,
      stack: event.error?.stack,
      timestamp: new Date().toISOString(),
    }),
  }).catch(err => {
    console.error('Failed to report error:', err);
  });
});

