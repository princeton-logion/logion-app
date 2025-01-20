const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const axios = require('axios');
const log = require('electron-log');

// Set log file location
log.transports.file.resolvePath = () => path.join(app.getPath('userData'), 'logs', 'logion-app.log');
const logFilePath = log.transports.file.getFile().path;

let backendProcess;
let loadingWindow;
let mainWindow;

function createLoadingWindow() {
    loadingWindow = new BrowserWindow({
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

    loadingWindow.loadURL(`file://${path.join(__dirname, 'loading.html')}`);

    loadingWindow.on('closed', () => {
        loadingWindow = null;
    });
    loadingWindow.webContents.on('did-finish-load', () => {
        log.info('Loading window loaded')
    })
}

function createMainWindow() {
    mainWindow = new BrowserWindow({
        width: 1000,
        height: 650,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
        },
    });

    mainWindow.loadURL(`file://${path.join(__dirname, 'frontend', 'index.html')}`);

    mainWindow.webContents.on('did-finish-load', () => {
        log.info('Main window loaded');
    });

    mainWindow.on('closed', () => {
        if (backendProcess) {
            backendProcess.kill();
            log.info('Backend process killed');
        }
        mainWindow = null;
    });
}

function startBackend() {
    let backendPath;

    // Determine the platform and set the backend path
    if (process.platform === 'win32') {
        backendPath = path.join(process.resourcesPath, 'extraResources', 'main.exe'); // Windows executable
    } else if (process.platform === 'darwin') {
        backendPath = path.join(process.resourcesPath, 'extraResources', 'main'); // macOS executable
    } else if (process.platform === 'linux') {
        backendPath = path.join(process.resourcesPath, 'extraResources', 'main'); // Linux executable
    } else {
        log.error('Unsupported platform');
        app.quit();
        return;
    }

    log.info('Resources Path:', process.resourcesPath);
    log.info('Backend Path:', backendPath);

    try {
        backendProcess = spawn(backendPath, [], {
            stdio: ['pipe', 'pipe', 'pipe'],
            env: { ...process.env, LOGION_LOG_PATH: logFilePath }, // Add LOGION_LOG_PATH
        });
        log.info('Backend process started');
    } catch (err) {
        log.error(`Failed to start backend: ${err.message}`);
    }

    backendProcess.on('spawn', () => {
        log.info('Backend Process has spawned');
    });

    backendProcess.on('error', (err) => {
        log.error(`Failed to start backend: ${err}`);
    });

    backendProcess.stdout.on('data', (data) => {
        log.info(`Backend STDOUT: ${data}`);
    });

    backendProcess.stderr.on('data', (data) => {
        log.error(`Backend STDERR: ${data}`);
    });

    backendProcess.on('close', (code) => {
        log.info(`Backend process exited with code ${code}`);
    });
}


async function waitForBackendReady() {
    const healthEndpoint = 'http://localhost:8000/health';
    const retryInterval = 500; // 500 ms
    const maxRetries = 200; // Retry for 10 seconds (20 * 500ms)

    await new Promise(resolve => setTimeout(resolve, 2000)); // wait 2 seconds before checking for health
    
    for (let i = 0; i < maxRetries; i++) {
        try {
            const response = await axios.get(healthEndpoint);
            if (response.status === 200) {
                log.info('Backend is ready');
                return true;
            }
        } catch (error) {
            log.info('Waiting for backend to be ready...');
        }
        await new Promise(resolve => setTimeout(resolve, retryInterval));
    }
    log.error('Backend failed to start within the timeout period');
    return false;
}

app.whenReady().then(async () => {
    createLoadingWindow(); // Show the loading screen
    startBackend(); // Start the backend

    const isBackendReady = await waitForBackendReady();

    if (isBackendReady) {
        loadingWindow.close(); // Close the loading screen
        createMainWindow(); // Open the main application window
    } else {
        log.error('Backend did not start. Exiting application.');
        app.quit();
    }
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

ipcMain.handle('predict-request', async (event, requestData) => {
    try {
        const response = await axios.post('http://localhost:8000/prediction', requestData);
        return response.data;
    } catch (error) {
        log.error("Error processing predict-request:", error);
        throw new Error(error.message); // Re-throw the error for the frontend
    }
});

ipcMain.handle('detect-request', async (event, requestData) => {
    try {
        const response = await axios.post('http://localhost:8000/detection', requestData);
        return response.data;
    } catch (error) {
        log.error("Error processing detect-request:", error);
        throw new Error(error.message); // Re-throw the error for the frontend
    }
});