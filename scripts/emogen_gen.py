import sys
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from dotenv import load_dotenv
import os
from tqdm import tqdm

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
        transforms.Resize((380, 380)),  # Adjust the image size to match EfficientNet-B4 input size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the access token from the .env file
    load_dotenv()
    access_token = os.getenv("ACCESS_TOKEN")

    # Load Stable Diffusion XL model
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"  # Updated model identifier
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler", use_auth_token=access_token)
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32, added_cond_kwargs={})
    pipe.safety_checker = None  # Disable the safety checker
    pipe = pipe.to("cpu")

    # Load and preprocess the input image
    input_image = Image.open(input_image_path).convert("RGB")
    input_image_resized = input_image.resize((768, 768))  # Resize the input image to match the model's expected size

    # Generate the prompt based on the desired emotion
    prompt = f"The original image slightly modified to convey {desired_emotion} while keeping the overall content and design similar."

    # Initialize the image variable before the loop
    image = None

    # Generate the image using Stable Diffusion XL
    generator = torch.Generator("cpu").manual_seed(1024)
    max_iterations = 10
    progress_bar = tqdm(total=max_iterations, desc="Generating Image", unit="iteration")
    for i in range(max_iterations):
        try:
            result = pipe(prompt, num_inference_steps=25, guidance_scale=7.5, generator=generator, init_image=input_image_resized, callback=progress_callback, callback_steps=1)

            if result is None or not hasattr(result, 'images') or len(result.images) == 0:
                raise ValueError("Generated image is None or 'images' key is missing")

            image = result.images[0]

            # Preprocess the generated image for emotion detection
            image_tensor = transform(image).unsqueeze(0).to("cpu")

            # Detect the emotion in the generated image
            with torch.no_grad():
                emotion_outputs = emotion_model(image_tensor)
                _, predicted_emotion_idx = torch.max(emotion_outputs, 1)
                predicted_emotion = unique_emotions[predicted_emotion_idx.item()]

            progress_bar.set_postfix({"Detected Emotion": predicted_emotion})

            if predicted_emotion == desired_emotion:
                progress_bar.close()
                print(f"Desired emotion '{desired_emotion}' detected in the generated image.")
                break

            if i == max_iterations - 1:
                progress_bar.close()
                print(f"Maximum iterations reached. The generated image may not strongly convey the desired emotion.")
        except UnidentifiedImageError as e:
            print(f"Error loading image: {str(e)}")
            continue
        except Exception as e:
            print(f"Error in iteration {i+1}: {str(e)}")
            continue

    return image

def progress_callback(step: int, timestep: int, latents: torch.FloatTensor) -> None:
    progress_bar.update(1)

# Get the input image path, desired emotion, and model path from command-line arguments
input_image_path = sys.argv[1]
desired_emotion = sys.argv[2]
model_path = sys.argv[3]

# Pass the model path as an argument to the generate_image_with_emotion function
output_image = generate_image_with_emotion(input_image_path, desired_emotion, model_path)

# Save the generated image
if output_image is not None:
    output_image_path = f"EmoGen/outputs/generated_image_{desired_emotion}.jpg"
    output_image.save(output_image_path)
    print(f"Generated image saved at: {output_image_path}")
else:
    print("No image generated.")