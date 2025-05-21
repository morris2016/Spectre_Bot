import React, { createContext, useContext } from 'react';

export const WebSocketContext = createContext({
  connect: () => {},
  disconnect: () => {}
});

export const WebSocketProvider = ({ children }) => {
  const value = {
    connect: () => {},
    disconnect: () => {}
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocketContext = () => useContext(WebSocketContext);
