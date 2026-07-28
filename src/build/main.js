const { app, BrowserWindow, screen, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const axios = require('axios');
const log = require('electron-log');
const fs = require('fs');
const dotenv = require('dotenv');
const readline = require('node:readline');

// set log file
log.transports.file.resolvePathFn = () => path.join(app.getPath('userData'), 'logs', 'logion-app.log');
const logFilePath = log.transports.file.getFile().path;

function getEnvVars() {
    // find app.env file + load
    const envPath = app.isPackaged 
        ? path.join(process.resourcesPath, 'app.env') 
        : path.join(__dirname, 'app.env');
    
    let vars = { ...process.env };

    if (fs.existsSync(envPath)) {
        log.info(`Loading app.env from ${envPath}`);
        const parsed = dotenv.parse(fs.readFileSync(envPath));
        vars = { ...vars, ...parsed };
    } else {
        log.info('app.env file not found, using system env only');
    }

    if (vars.LOGION_RESOURCES_CONFIG) {
        log.info(`Remote Config Source: ${vars.LOGION_RESOURCES_CONFIG}`);
    }

    return vars;
}
const appEnv = getEnvVars();
const HOST = appEnv.LOGION_HOST || '127.0.0.1';
let baseUrl = null;

let backendProcess;
let loadingWindow;
let mainWindow;

// avoid multiple Logion instances
if (!app.requestSingleInstanceLock()) {
    app.quit();
} else {
    app.on('second-instance', () => {
        if (mainWindow) {
            if (mainWindow.isMinimized()) mainWindow.restore();
            mainWindow.focus();
        }
    });
}

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

    const startupURL = baseUrl;

    log.info(`[createMainWindow] Loading URL: ${startupURL}`);
        mainWindow.loadURL(startupURL);

    mainWindow.webContents.on('did-finish-load', () => {
        log.info('Main app window loaded');
    });

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

function startBackend() {
    let backendPath;
    let staticDir;

    const isDev = process.env.NODE_ENV === 'development';

    if (isDev) {
        backendPath = DEV_PATH;
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
            // kill(-pid) reaches descendants with own process group on linx/mac
            detached: process.platform !== 'win32',
            env: { ...appEnv, STATIC_DIR: staticDir, LOGION_LAUNCHER: 'electron', PYTHONUNBUFFERED: '1' },
        });
        log.info('Backend API started.');
    } catch (err) {
        log.error(`Unable to spawn API server: ${err.message}`);
        return Promise.reject(err);
    }

    backendProcess.on('spawn', () => {
        log.info('Backend API spawned.');
    });

    backendProcess.on('error', (err) => {
        log.error(`Unable to spawn API server: ${err}`);
    });

    backendProcess.stderr.on('data', (data) => {
        log.info(`API: ${data}`);
    });

    backendProcess.on('close', (code) => {
        log.info(`Quit API with code ${code}`);
    });

    // wait for backend port report
    return new Promise((resolve, reject) => {
        const timer = setTimeout(
            () => reject(new Error('No reported port from API')),
            120000
        );

        readline.createInterface({ input: backendProcess.stdout })
            .on('line', (line) => {
                log.info(`API STDOUT: ${line}`);
                const m = /^__LOGION_PORT__=(\d+)$/.exec(line);
                if (m) {
                    clearTimeout(timer);
                    resolve(Number(m[1]));
                }
            });

        backendProcess.on('exit', (code) => {
            clearTimeout(timer);
            reject(new Error(`API exit during startup: ${code}`));
        });
    });
}


// terminate backend + all processes
function stopBackend() {
    if (!backendProcess) return;
    const proc = backendProcess;
    backendProcess = null;

    log.info(`Terminating backend (pid ${proc.pid})...`);
    try {
        if (process.platform === 'win32') {
            spawn('taskkill', ['/pid', String(proc.pid), '/T', '/F']);
        } else {
            process.kill(-proc.pid, 'SIGTERM');
            setTimeout(() => {
                try { process.kill(-proc.pid, 'SIGKILL'); } catch {}
            }, 5000);
        }
    } catch (err) {
        log.error(`Unable to terminate backend: ${err.message}`);
    }
}


// check health endpoint for API server
async function wait4ServerReady() {
    const healthEndpoint = `${baseUrl}/health`;
    const retryInterval = 500; // 500 ms
    const maxRetries = 240; // wait up 2 mins (for slow Win 1st open)

    for (let i = 0; i < maxRetries; i++) {
        try {
            const response = await axios.get(healthEndpoint, { timeout: 2000 });
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
    createLoadingWindow();

    let port;
    try {
        port = await startBackend();
    } catch (err) {
        log.error(`Backend startup failed: ${err.message}`);
        dialog.showErrorBox('Logion unable to start', err.message);
        stopBackend();
        app.quit();
        return;
    }

    baseUrl = `http://${HOST}:${port}`;
    log.info(`Backend port ${port}, base URL ${baseUrl}`);

    const isBackendReady = await wait4ServerReady();

    if (isBackendReady) {
        if (loadingWindow) loadingWindow.close();
        createMainWindow();
    } else {
        log.error('Unable to start API server. Quit app.');
        dialog.showErrorBox('Unable to start Logion. Server failed to start.');
        stopBackend();
        app.quit();
    }
});


// kill server when app quits
app.on('before-quit', stopBackend);
app.on('will-quit', stopBackend);
process.on('exit', stopBackend);

// quit app
app.on('window-all-closed', () => {
     app.quit();
});
