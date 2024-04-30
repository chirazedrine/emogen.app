const { app, BrowserWindow } = require('electron');
const path = require('path');
const { ipcMain } = require('electron');
const { spawn } = require('child_process');

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  win.loadFile(path.join(__dirname, '../public/index.html'));
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

ipcMain.handle('run-script', async (event, { script, emotion, imagePath }) => {
  console.log('Received run-script request:', { script, emotion, imagePath });

  return new Promise((resolve, reject) => {
    const appPath = app.getAppPath();
    const scriptPath = path.join(appPath, 'scripts', `${script === 'emotion-detection' ? 'emogen_det.py' : 'emogen_gen.py'}`);
    const modelPath = path.join(__dirname, '../models/best_model_v2.pth');

    // Execute the script using Python
    const args = [scriptPath, imagePath];
    if (script === 'emotion-generation') {
      args.push(emotion, modelPath);
    } else {
      args.push(modelPath);
    }
    const pythonProcess = spawn('python', args);

    let outputData = '';
    let progressData = '';

    pythonProcess.stdout.on('data', (data) => {
      const dataString = data.toString();
      if (dataString.includes('|') && dataString.includes('%')) {
        // Progress bar output
        progressData += dataString;
        console.log('Progress:', progressData.trim());
      } else {
        // Regular output
        outputData += dataString;
        console.log('Received data from Python script:', dataString.trim());
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error('Error from Python script:', data.toString());
    });

    pythonProcess.on('close', (code) => {
      console.log(`Python script exited with code ${code}`);

      if (code === 0) {
        const result = outputData.trim();
        resolve(result);
      } else {
        reject(new Error('Python script encountered an error'));
      }
    });

    pythonProcess.stdout.on('data', (data) => {
      if (script === 'emotion-generation') {
        // Image data received from the Python script
        resolve(data);
      } else {
        outputData += data.toString();
      }
    });
  });
});
