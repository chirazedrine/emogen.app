<p align="center">
  <img src="assets/icon.png" alt="Emogen App Logo" width="200">
</p>

# EmoGen

EmoGen is an emotion detection application built with Electron, Python, and PyTorch. It leverages state-of-the-art deep learning models to analyze images and determine the closest emotion associated with it, amongst 8 emotions (amusement, anger, awe, contentment, disgust, excitement, fear and sadness).

## Installation

1. Clone the repository:

```
git clone https://github.com/your-username/emogen-app.git
```

2. Navigate to the project directory:

```
cd emogen-app
```

3. Install the necessary dependencies:

```
npm install
pip install -r requirements.txt
```

4. Run the application:

```
npm start
```

## Methodology and models

### Emotion detection

The emotion detection functionality in EmoGen is powered by the EfficientNet-B4 model, a state-of-the-art convolutional neural network (CNN) architecture. The EfficientNet-B4 model is pre-trained on the ImageNet dataset and fine-tuned for emotion detection using a custom dataset.

The emotion detection process involves the following steps:

1. Image Preprocessing: The input image is resized to (456, 456) pixels to match the input size of the EfficientNet-B4 model. It is then normalized using mean and standard deviation values specific to the ImageNet dataset.

2. Model Inference: The preprocessed image is passed through the fine-tuned EfficientNet-B4 model, which outputs a probability distribution over the 8 predefined emotions.

3. Emotion Prediction: The emotion with the highest probability is selected as the detected emotion for the input image.

### Emotion generation

The emotion generation functionality in EmoGen utilizes the Stable Diffusion XL model, a powerful text-to-image generation model. The Stable Diffusion XL model is capable of generating realistic images conditioned on a given text prompt.

The emotion generation process involves the following steps:

1. Prompt Generation: Based on the desired emotion, a text prompt is generated to guide the image generation process. The prompt aims to slightly modify the original image to convey the desired emotion while preserving the overall content and design.

2. Image Generation: The Stable Diffusion XL model takes the generated prompt and the original image as inputs. It iteratively refines the image using a diffusion-based approach, guided by the prompt and a guidance scale parameter.

3. Emotion Verification: After each iteration, the generated image is passed through the emotion detection model to assess if it successfully conveys the desired emotion. If the desired emotion is detected, the generation process is considered successful.

4. Iteration and Refinement: If the desired emotion is not detected, the generation process continues for a maximum number of iterations or until the desired emotion is achieved. This iterative refinement helps ensure that the generated image accurately conveys the intended emotion.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

The following open-source libraries and resources that made this project possible:

- [Electron](https://www.electronjs.org/)
- [Python](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [StableDiffusionXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9)
