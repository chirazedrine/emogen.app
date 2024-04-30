import sys
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from dotenv import load_dotenv
import os

def generate_image_with_emotion(input_image_path, desired_emotion, model_path):
    # Load the emotion detection model
    emotion_model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
    num_ftrs = emotion_model.classifier[1].in_features
    unique_emotions = ['amusement', 'awe', 'anger', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
    emotion_model.classifier[1] = torch.nn.Linear(num_ftrs, len(unique_emotions))
    emotion_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    emotion_model.eval()

    # Image transformations for emotion detection
    transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the access token from the .env file
    load_dotenv()
    access_token = os.getenv("ACCESS_TOKEN")

    # Load Stable Diffusion 2.1 img2img model
    model_id = "stabilityai/stable-diffusion-2-1"
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler", use_auth_token=access_token)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32, use_auth_token=access_token)
    pipe = pipe.to("cpu")

    # Load and preprocess the input image
    input_image = Image.open(input_image_path).convert("RGB")
    input_image_resized = input_image.resize((512, 512))

    # Generate the prompt based on the desired emotion
    prompt = f"The original image with subtle changes to convey a sense of {desired_emotion}, while preserving the overall content, style, and composition."

    max_iterations = 1
    for i in range(max_iterations):
        # Generate the image using Stable Diffusion 2.1 img2img with adjusted parameters
        generator = torch.Generator("cpu").manual_seed(1024)
        image = pipe(prompt=prompt, image=input_image_resized, num_inference_steps=50, guidance_scale=7, strength=0.4, generator=generator).images[0]

        # Preprocess the generated image for emotion detection
        image_tensor = transform(image).unsqueeze(0).to("cpu")

        # Detect the emotion in the generated image
        with torch.no_grad():
            emotion_outputs = emotion_model(image_tensor)
            _, predicted_emotion_idx = torch.max(emotion_outputs, 1)
            predicted_emotion = unique_emotions[predicted_emotion_idx.item()]

        print(f"Iteration {i+1}: Detected Emotion - {predicted_emotion}")

        if predicted_emotion == desired_emotion:
            print(f"Desired emotion '{desired_emotion}' detected in the generated image.")
            break

        if i == max_iterations - 1:
            print(f"Maximum iterations reached. The generated image may not strongly convey the desired emotion.")

    return image

# Get the input image path, desired emotion, and model path from command-line arguments
input_image_path = sys.argv[1]
desired_emotion = sys.argv[2]
model_path = sys.argv[3]

# Pass the model path as an argument to the generate_image_with_emotion function
output_image = generate_image_with_emotion(input_image_path, desired_emotion, model_path)

# Save the generated image
if output_image is not None:
    # Get the initial image's name without the extension
    initial_image_name = os.path.splitext(os.path.basename(input_image_path))[0]
    
    # Create the output directory if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the base filename for the generated image
    base_filename = f"{initial_image_name}_{desired_emotion}"
    
    # Check if the filename already exists and add a number if necessary
    counter = 1
    output_image_path = os.path.join(output_dir, f"{base_filename}.jpg")
    while os.path.exists(output_image_path):
        output_image_path = os.path.join(output_dir, f"{base_filename}_{counter}.jpg")
        counter += 1
    
    # Save the generated image
    output_image.save(output_image_path)
    print(f"Generated image saved at: {output_image_path}")
    
    # Send the generated image data back to the renderer process
    with open(output_image_path, 'rb') as file:
        image_data = file.read()
        sys.stdout.buffer.write(image_data)
else:
    print("No image generated.")