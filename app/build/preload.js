    // preload.js
    const { contextBridge } = require('electron');
    require('dotenv').config();


    contextBridge.exposeInMainWorld('env', {
        REACT_APP_API_BASE_URL: process.env.REACT_APP_API_BASE_URL
    })