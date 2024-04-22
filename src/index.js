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
    const scriptPath = path.join(__dirname, `../scripts/${script === 'emotion-detection' ? 'emogen_det.py' : 'emogen_gen.py'}`);
    const modelPath = path.join(__dirname, '../models/best_model_v2.pth');

    // Execute the script using Python
    const pythonProcess = spawn('python', [scriptPath, imagePath, modelPath]);

    let outputData = '';
    pythonProcess.stdout.on('data', (data) => {
      console.log('Received data from Python script:', data.toString());
      outputData += data;
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error('Error from Python script:', data.toString());
    });

    pythonProcess.on('close', (code) => {
      console.log(`Python script exited with code ${code}`);

      if (code === 0) {
        resolve(outputData.trim());
      } else {
        reject(new Error('Python script encountered an error'));
      }
    });
  });
});