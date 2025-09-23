// preload.js
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electron', {
   //ipcRenderer: {
        //invoke: (channel, data) => ipcRenderer.invoke(channel, data),
    //}
    log: (level, message) => ipcRenderer.send('log-message', { level, message })
});

//contextBridge.exposeInMainWorld('env', {'http://localhost:8000'});
