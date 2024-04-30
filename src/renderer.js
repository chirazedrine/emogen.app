const { ipcRenderer } = require('electron');

new Vue({
  el: '#app',
  data: {
    selectedMode: null,
    selectedImage: null,
    detectionResult: null,
    selectedEmotion: 'amusement',
    generatedImage: null,
    imageUrl: null,
    isLoading: false,
  },
  methods: {
    selectMode(mode) {
      this.selectedMode = mode;
      this.resetResults();
    },
    onImageSelected(event) {
      this.selectedImage = event.target.files[0];
      this.imageUrl = URL.createObjectURL(this.selectedImage);
      this.resetResults();
    },
    async runDetection() {
      this.isLoading = true;
      try {
        const result = await ipcRenderer.invoke('run-script', {
          script: 'emotion-detection',
          imagePath: this.selectedImage.path,
        });
        this.detectionResult = result;
      } catch (error) {
        console.error('An error occurred:', error.message);
      } finally {
        this.isLoading = false;
      }
    },
    async runGeneration() {
      this.isLoading = true;
      try {
        const result = await ipcRenderer.invoke('run-script', {
          script: 'emotion-generation',
          emotion: this.selectedEmotion,
          imagePath: this.selectedImage.path,
        });
        this.generatedImage = result;
      } catch (error) {
        console.error('An error occurred:', error.message);
      } finally {
        this.isLoading = false;
      }
    },
    resetResults() {
      this.detectionResult = null;
      this.generatedImage = null;
    },
  },
});