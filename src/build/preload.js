// preload.js
const { contextBridge, ipcRenderer } = require('electron');
require('dotenv').config();

contextBridge.exposeInMainWorld('electron', {
    ipcRenderer: {
        invoke: (channel, data) => ipcRenderer.invoke(channel, data),
    }
});

//contextBridge.exposeInMainWorld('env', {'http://localhost:8000'});
