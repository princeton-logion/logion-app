const { app, BrowserWindow, screen, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const axios = require('axios');
const log = require('electron-log');

// set log file
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
    const { width, height } = screen.getPrimaryDisplay().workAreaSize;
    mainWindow = new BrowserWindow({
        width: width,
        height: height,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js')
        },
    });

    const startupURL = 'http://127.0.0.1:8000'; //dynamic load from env variables

    log.info(`[createMainWindow] Loading URL: ${startupURL}`);
        mainWindow.loadURL(startupURL);

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
    let staticDir;

    const isDev = process.env.NODE_ENV === 'development';

    if (isDev) {
        backendPath = '/Users/jm9095/logion-app/src/backend/dist/main';
        if (process.platform === 'win32') {
            backendPath += '.exe';
        }
        staticDir = path.join(__dirname, '..', 'frontend', 'build');
        log.info("Running in dev mode.");
    } else {
    if (process.platform === 'win32') {
        backendPath = path.join(process.resourcesPath, 'extraResources', 'main.exe'); // win exec
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
    
    staticDir = path.join(process.resourcesPath, 'frontend')}

    log.info('Resources Path:', process.resourcesPath);
    log.info('Backend API Path:', backendPath);
    log.info('Frontend Static Path:', staticDir)

    try {
        backendProcess = spawn(backendPath, [], {
            stdio: ['pipe', 'pipe', 'pipe'],
            env: { ...process.env, STATIC_DIR: staticDir },
        });
        log.info('Backend API started.');
    } catch (err) {
        log.error(`Unable to spawn API server: ${err.message}`);
    }

    backendProcess.on('spawn', () => {
        log.info('Backend API spawned.');
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
                log.info('Server ready.');
                return true;
            }
        } catch (error) {
            log.info('Awaiting server...');
        }
        await new Promise(resolve => setTimeout(resolve, retryInterval));
    }
    log.error('Unable to spawn server within timeout period.');
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
