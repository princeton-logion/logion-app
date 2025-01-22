const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
    send: (channel, data) => ipcRenderer.send(channel, data),
    handle: (channel, func) => ipcRenderer.on(channel, (event, ...args) => func(...args)),
    invoke: (channel, data) => ipcRenderer.invoke(channel, data),
    once: (channel, func) => ipcRenderer.once(channel, (event, ...args) => func(...args)),
});

ipcRenderer.on('set-axios-base-url', (event, baseURL) => {
    window.axios.defaults.baseURL = baseURL;
});