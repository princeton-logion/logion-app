import React, { createContext, useState, useEffect, useRef, useContext, useCallback } from 'react';

// create unique user ID
function genUID() {
    return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
        (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
    );
}

// backend Websocket relative path
/* const getWebSocketURL = () => {
    const host = window.location.host;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const path = '/ws/';
    return `${protocol}//${host}${path}`;
}; */
const getWebSocketURL = () => {
    const host = window.location.host;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // get base path from browser url; OOD: "/node/della-[host]/[port]/", local: "/"
    let basePath = window.location.pathname;
    // if base path doesn't end in "/", add "/"
    if (!basePath.endsWith('/')) {
        basePath += '/';
    }
    // append ws endpoint to base path
    const path = `${basePath}ws/`; 

    return `${protocol}//${host}${path}`;
};

// establish WS context
const WebSocketContext = createContext({
    isConnected: false,
    clientId: null,
    sendMessage: (message) => {},
    addMessageHandler: (handler) => {},
    removeMessageHandler: (handler) => {}
});

export const useWebSocket = () => useContext(WebSocketContext);

export const WebSocketProvider = ({ children }) => {
    const [isConnected, setIsConnected] = useState(false);
    const [isConnecting, setIsConnecting] = useState(false);
    const websocket = useRef(null);
    const clientId = useRef(genUID()); // one user ID per runtime
    const messageHandlers = useRef(new Set());

    const addMessageHandler = useCallback((handler) => {
        messageHandlers.current.add(handler);
        console.log("Created message handler. Number of handlers:", messageHandlers.current.size);
    }, []);

    const removeMessageHandler = useCallback((handler) => {
        messageHandlers.current.delete(handler);
        console.log("Deleted message handler. Number of handlers:", messageHandlers.current.size);
    }, []);


    // function to create WS connection
    const connect = useCallback(() => {
        // handle multiple potential connections -- prevent simultaneous connection
        if ((websocket.current && websocket.current.readyState === WebSocket.OPEN) || isConnecting) {
             console.log("Already connected. Ignorint connection attempt.");
            return;
        }

        const dynamicUrl = `${getWebSocketURL()}${clientId.current}`;
        console.log(`Attempting WebSocket connection to ${dynamicUrl}`);
        setIsConnecting(true);
    
        // Use the dynamically generated URL
        const ws = new WebSocket(dynamicUrl);

        ws.onopen = () => {
            console.log("WebSocket connected");
            websocket.current = ws;
            setIsConnected(true);
            setIsConnecting(false);
        };

        ws.onclose = (event) => {
            console.log("WebSocket disconnected:", event.code, event.reason);
            // clear ref
            if (websocket.current === ws) {
                websocket.current = null;
            }
            setIsConnected(false);
            setIsConnecting(false);
        };

        ws.onerror = (error) => {
            console.error("WebSocket error:", error);
            if (websocket.current === ws) {
                websocket.current = null;
            }
            setIsConnected(false);
            setIsConnecting(false);
        };

        ws.onmessage = (event) => {
             console.log("Message received:", event.data);
             try {
                const message = JSON.parse(event.data);
                messageHandlers.current.forEach(handler => {
                    try {
                       handler(message);
                    } catch (handlerError) {
                       console.error("Message handler error:", handlerError);
                    }
                });
             } catch (error) {
                console.error("Unable to read WebSocket message:", error, "Data:", event.data);
            }
        };
    }, []);


    // connect on mount
    useEffect(() => {
        console.log("[WebSocketProvider] Mount/connect effect triggered. Calling connect().");
        connect();

        // cleanup on unmount
        return () => {
            console.log("WebSocket cleanup. Disconnecting.");
            if (websocket.current) {
                websocket.current.onopen = null;
                websocket.current.onmessage = null;
                websocket.current.onerror = null;
                websocket.current.onclose = null;
                websocket.current.close();
                websocket.current = null;
            }
            messageHandlers.current.clear();
            setIsConnected(false);
            setIsConnecting(false);
        };
    }, [connect]);


    // function for messages, with connection checks
    const sendMessage = useCallback((messageObject) => {
        if (websocket.current && websocket.current.readyState === WebSocket.OPEN) {
            try {
                const messageString = JSON.stringify(messageObject);
                websocket.current.send(messageString);
            } catch (error) {
                console.error("Unable to send WebSocket message:", error);
            }
        } else {
            console.error("Unable to send Websocket message. Not connected.");
        }
    }, []); 
    
    const contextValue = {
        isConnected,
        // websocket: websocket.current,
        clientId: clientId.current,
        sendMessage,
        addMessageHandler,
        removeMessageHandler
    };

    return (
        <WebSocketContext.Provider value={contextValue}>
            {children}
        </WebSocketContext.Provider>
    );
};