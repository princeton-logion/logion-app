const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const axios = require('axios');
const log = require('electron-log');

// set log file -- same as backend
log.transports.file.resolvePath = () => path.join(app.getPath('userData'), 'logs', 'logion-app.log');
const logFilePath = log.transports.file.getFile().path;

let backendProcess;
let loadingAPIscreen;
let AppMainWindow;

function createLoadingScreen() {
    loadingAPIscreen = new BrowserWindow({
        width: 400,
        height: 300,
        frame: false,
        transparent: true,
        alwaysOnTop: false,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
        },
    });

    loadingAPIscreen.loadURL(`file://${path.join(__dirname, 'loading.html')}`);

    loadingAPIscreen.on('closed', () => {
        loadingAPIscreen = null;
    });
    loadingAPIscreen.webContents.on('did-finish-load', () => {
        log.info('Loading window loaded')
    })
}

function createMainWindow() {
    AppMainWindow = new BrowserWindow({
        width: 1000,
        height: 650,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
        },
    });

    AppMainWindow.loadURL(`file://${path.join(__dirname, 'frontend', 'index.html')}`);

    AppMainWindow.webContents.on('did-finish-load', () => {
        log.info('Main window loaded.');
    });

    AppMainWindow.on('closed', () => {
        if (backendProcess) {
            backendProcess.kill();
            log.info('Killed backend API process.');
        }
        AppMainWindow = null;
    });
}

function startBackend() {
    let backendPath;

    const isDevelopment = process.env.NODE_ENV === 'development';

    if (isDevelopment) {
        backendPath = '/Users/jm9095/logion-app/src/backend/dist';
        if (process.platform === 'win32') {
            backendPath += '.exe';
        }
        log.info("Running in developer mode.");
    } else if (process.platform === 'win32') {
        backendPath = path.join(process.resourcesPath, 'extraResources', 'main.exe'); // Win exec
        log.info("Running production application on Windows.");
    } else if (process.platform === 'darwin') {
        backendPath = path.join(process.resourcesPath, 'extraResources', 'main'); // macOS exec
        log.info("Running production application on macOS.");
    } else if (process.platform === 'linux') {
        backendPath = path.join(process.resourcesPath, 'extraResources', 'main'); // linx exec
        log.info("Running production application on Linux.");
    } else {
        log.error('Unsupported platform. Use one of these supported platforms: Windows, macOS, Linux.');
        app.quit();
        return;
    }

    log.info('Resources Path:', process.resourcesPath);
    log.info('Backend API Path:', backendPath);

    try {
        backendProcess = spawn(backendPath, [], {
            stdio: ['pipe', 'pipe', 'pipe'],
            env: { ...process.env, LOGION_LOG_PATH: logFilePath }, // add LOGION_LOG_PATH
        });
        log.info('Backend API started.');
    } catch (err) {
        log.error(`Failed to start API server: ${err.message}`);
    }

    backendProcess.on('spawn', () => {
        log.info('Backend API has spawned.');
    });

    backendProcess.on('error', (err) => {
        log.error(`Failed to start API server: ${err}`);
    });

    backendProcess.stdout.on('data', (data) => {
        log.info(`API STDOUT: ${data}`);
    });

    backendProcess.stderr.on('data', (data) => {
        log.error(`API STDERR: ${data}`);
    });

    backendProcess.on('close', (code) => {
        log.info(`Exited API with code ${code}`);
    });
}


// 
async function wait4ServerReady() {
    const healthEndpoint = 'http://127.0.0.1:8000/health';
    const retryInterval = 500; // 500 ms
    const maxRetries = 100; // retry for 10 seconds (20 * 500ms)

    for (let i = 0; i < maxRetries; i++) {
        try {
            const response = await axios.get(healthEndpoint);
            if (response.status === 200) {
                log.info('API server ready.');
                return true;
            }
        } catch (error) {
            log.info('Waiting for API server...');
        }
        await new Promise(resolve => setTimeout(resolve, retryInterval));
    }
    log.error('API server failed to start within the timeout period.');
    return false;
}

app.whenReady().then(async () => {
    createLoadingScreen(); // show loading screen on startup
    startBackend(); // start backend on startup

    const isBackendReady = await wait4ServerReady();

    if (isBackendReady) {
        loadingAPIscreen.close(); // close loading screen when server is ready 
        createMainWindow(); // then open main window
    } else {
        log.error('API server failed to start. Exiting application.');
        app.quit();
    }
});

// quit app
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

// prediction page
ipcMain.handle('predict-request', async (event, requestData) => {
    try {
        const response = await axios.post('http://127.0.0.1:8000/prediction', requestData);
        return response.data;
    } catch (error) {
        log.error("Error processing predict-request IPC:", error);
    }
});

// error detection page
ipcMain.handle('detect-request', async (event, requestData) => {
    try {
        const response = await axios.post('http://127.0.0.1:8000/detection', requestData);
        return response.data;
    } catch (error) {
        log.error("Error processing detect-request IPC:", error);
    }
});
