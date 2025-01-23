import torch
from PIL import Image
from torchvision import transforms, models
import pandas as pd

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load car names from names.csv without using the first row as a header
car_names_file = 'names.csv'
car_names = pd.read_csv(car_names_file, header=None, names=["Model"])
car_labels = car_names["Model"].sort_values().tolist()

# Load the EfficientNet-B3 model without downloading the weights
car_model_classifier = models.efficientnet_b3()

# Load the state dictionary from the local weights file
weights_path = r"efficientnet_b3_rwightman-b3899882.pth"
state_dict = torch.load(weights_path, map_location=device)
car_model_classifier.load_state_dict(state_dict)

# Redefine the classifier to match the architecture used during training
num_features = car_model_classifier.classifier[1].in_features
car_model_classifier.classifier = torch.nn.Sequential(
    torch.nn.Linear(num_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),  # Dropout to prevent overfitting
    torch.nn.Linear(512, 196)  # 196 output classes
)
car_model_classifier = car_model_classifier.to(device)
car_model_classifier.eval()  # Set to evaluation mode

# Load the custom model weights
car_model_classifier.load_state_dict(torch.load("best_stanford_cars_model.pth", map_location=device))

# Define a function to classify the vehicle image
def classify_vehicle(image):
    # Define the image transformations (resize, normalize, etc.)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for ImageNet
    ])
    
    # Apply transformations to the image
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = car_model_classifier(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get the top 5 predictions
        top5_probabilities, top5_indices = probabilities.topk(5, 1, largest=True, sorted=True)
        
        # Map the predictions to their corresponding labels
        predictions = [car_labels[idx] for idx in top5_indices[0]]

    return predictions

