const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const axios = require('axios');
const log = require('electron-log');

// set log file -- same as backend
log.transports.file.resolvePathFn = () => path.join(app.getPath('userData'), 'logs', 'logion-app.log');
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
        log.info('Main app window loaded');
    });

    mainWindow.on('closed', () => {
        if (backendProcess) {
            backendProcess.kill();
            log.info('Quit backend API');
        }
        mainWindow = null;
    });
}

function startBackend() {
    let backendPath;

    const isDev = process.env.NODE_ENV === 'development';

    if (isDev) {
        backendPath = '/Users/jm9095/logion-app/src/backend/dist/main';
        if (process.platform === 'win32') {
            backendPath += '.exe';
        }
        log.info("Running in dev mode.");
    } else if (process.platform === 'win32') {
        backendPath = path.join(process.resourcesPath, 'extraResources', 'main.exe'); // Win exec
        log.info("Running prod app on Windows.");
    } else if (process.platform === 'darwin') {
        backendPath = path.join(process.resourcesPath, 'extraResources', 'main'); // macOS exec
        log.info("Running prod app on macOS.");
    } else if (process.platform === 'linux') {
        backendPath = path.join(process.resourcesPath, 'extraResources', 'main'); // linx exec
        log.info("Running prod app on Linux.");
    } else {
        log.error('Invalid platform. Valid platforms: Windows, macOS, Linux.');
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
        log.error(`Unable to spawn API server: ${err.message}`);
    }

    backendProcess.on('spawn', () => {
        log.info('Backend API has spawned.');
    });

    backendProcess.on('error', (err) => {
        log.error(`Unable to spawn API server: ${err}`);
    });

    backendProcess.stdout.on('data', (data) => {
        log.info(`API STDOUT: ${data}`);
    });

    backendProcess.stderr.on('data', (data) => {
        log.error(`API STDERR: ${data}`);
    });

    backendProcess.on('close', (code) => {
        log.info(`Quit API with code ${code}`);
    });
}


// check health endpoint for API server
async function wait4ServerReady() {
    const healthEndpoint = 'http://127.0.0.1:8000/health';
    const retryInterval = 500; // 500 ms
    const maxRetries = 120; // wait up to 1 min

    for (let i = 0; i < maxRetries; i++) {
        try {
            const response = await axios.get(healthEndpoint);
            if (response.status === 200) {
                log.info('API server ready.');
                return true;
            }
        } catch (error) {
            log.info('Awaiting API server...');
        }
        await new Promise(resolve => setTimeout(resolve, retryInterval));
    }
    log.error('Unable to spawn API server within timeout period.');
    return false;
}

app.whenReady().then(async () => {
    createLoadingWindow(); // display load screen on launch
    startBackend(); // start backend on launch

    const isBackendReady = await wait4ServerReady();

    if (isBackendReady) {
        loadingWindow.close(); // close loading screen when API server ready
        createMainWindow(); // open main window
    } else {
        log.error('Unable to start API server. Quit app.');
        app.quit();
    }
});

// quit app
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

// word prediction page
ipcMain.handle('predict-request', async (event, requestData) => {
    try {
        const response = await axios.post('http://127.0.0.1:8000/prediction', requestData);
        return response.data;
    } catch (error) {
        log.error("Unable to process predict-request IPC:", error);
    }
});

// error detection page
ipcMain.handle('detect-request', async (event, requestData) => {
    try {
        const response = await axios.post('http://127.0.0.1:8000/detection', requestData);
        return response.data;
    } catch (error) {
        log.error("Unable to process detect-request IPC:", error);
    }
});
