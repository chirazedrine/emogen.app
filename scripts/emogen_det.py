import sys
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import efficientnet_b4

# Get the input image path and model path from command-line arguments
image_path = sys.argv[1]
model_path = sys.argv[2]

# Define the list of unique emotions (in the same order as during training)
unique_emotions = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']

# Define the transformations for the input image
transform = transforms.Compose([
    transforms.Resize((456, 456)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the saved model
model = efficientnet_b4()
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, len(unique_emotions))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Set up the device
device = torch.device('cpu')
model = model.to(device)

# Load and preprocess the input image
image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

# Perform emotion detection
with torch.no_grad():
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)

emotion_idx = predicted.item()
detected_emotion = unique_emotions[emotion_idx]

print(detected_emotion)