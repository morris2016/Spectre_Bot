

/**
 * QuantumSpectre Elite Trading System
 * UI Entry Point
 * 
 * This file serves as the entry point for the React application,
 * rendering the main App component to the DOM.
 */

import React from 'react';
import ReactDOM from 'react-dom';
import { Provider } from 'react-redux';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { BrowserRouter } from 'react-router-dom';
import { SnackbarProvider } from 'notistack';

import App from './App';
import store from './store';
import { theme } from './theme';
import { WorkspaceProvider } from './contexts/WorkspaceContext';
import { AuthProvider } from './contexts/AuthContext';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { VoiceAdvisorProvider } from './contexts/VoiceAdvisorContext';
import './i18n'; // Internationalization setup
import './index.css';

// Initialize performance monitoring
import './utils/performance';

// Setup error boundaries and logging
import './utils/errorHandler';

// Register Service Worker for PWA support
import { registerSW } from './serviceWorker';

// Prepare GPU acceleration if available
import { initializeGPUAcceleration } from './utils/gpuAcceleration';

/**
 * Initialize GPU Acceleration for UI rendering and calculations
 * Optimized for RTX 3050 with 8GB VRAM
 */
initializeGPUAcceleration();

/**
 * Main application rendering
 * Wrapped with all required providers:
 * - Redux store for state management
 * - Theme provider for consistent styling
 * - Router for navigation
 * - Workspace context for layout management
 * - Auth context for authentication
 * - WebSocket context for real-time data
 * - Voice advisor context for AI voice notifications
 * - Snackbar provider for notifications
 */
ReactDOM.render(
  
    
      
         {/* MUI CSS baseline normalization */}
        
          
            
              
                
                  
                    
                  
                
              
            
          
        
      
    
  ,
  document.getElementById('root')
);

// Register service worker for offline capabilities and PWA
registerSW();

// Add performance mark for initial render
performance.mark('app-rendered');

