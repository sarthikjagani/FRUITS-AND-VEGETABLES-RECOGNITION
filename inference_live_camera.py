import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import argparse
import warnings
import cv2
import numpy as np
# For a cleaner transformation pipeline, it's good practice to use transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

warnings.filterwarnings("ignore")

def make_args():
    parser = argparse.ArgumentParser("Test Argument")
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--checkpoint_path", "-c", type=str, default="trained_models/vegetables/best.pt")
    args = parser.parse_args()
    return args

def inference(args):
    # This is more portable across different machines (CPU, Nvidia GPU, Apple GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- FIX 1: Use all 36 categories from your training dataset ---
    # The order must be IDENTICAL to the one in your dataset.py file.
    categories = [
        "apple", "turnip", "chilli pepper", "raddish", "bell pepper", "pear", 
        "sweetpotato", "pomegranate", "peas", "capsicum", "spinach", "lettuce", 
        "kiwi", "lemon", "onion", "cauliflower", "potato", "jalepeno", 
        "sweetcorn", "cucumber", "paprika", "watermelon", "mango", "cabbage", 
        "grapes", "beetroot", "eggplant", "corn", "soy beans", "banana", 
        "ginger", "garlic", "pineapple", "tomato", "orange", "carrot"
    ]
    num_classes = len(categories)

    # Load the model architecture
    model = resnet50(weights=None)
    
    # --- FIX 2: Set the output features to 36, matching the checkpoint ---
    model.fc = nn.Linear(in_features=2048, out_features=num_classes)
    
    # Load the checkpoint with the trained weights
    # Using map_location ensures the model loads correctly regardless of the device it was saved on
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    softmax = nn.Softmax(dim=1)

    # --- Optional Improvement: Use torchvision.transforms for cleaner code ---
    transform = Compose([
        ToTensor(), # Converts numpy array (H, W, C) to tensor (C, H, W) and scales to [0, 1]
        Resize((args.image_size, args.image_size)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(0)
    window_name = 'Live Camera Feed'

    with torch.no_grad():
        while cap.isOpened():
            flag, frame = cap.read()
            if not flag:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_tensor = transform(image_rgb).unsqueeze(0)
            image_tensor = image_tensor.to(device)

            prediction = model(image_tensor)
            prob = softmax(prediction)
            max_value, max_index = torch.max(prob, dim=1)
            
            predicted_category = categories[max_index.item()]
            confidence = max_value.item()

            text = f"{predicted_category} ({confidence:.4f})"
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(window_name, frame) # Use the window name variable

            key_press = cv2.waitKey(1) & 0xFF

            # --- IMPROVEMENT: Check for 'q' key press ---
            if key_press == ord('q'):
                print("'q' pressed, closing window.")
                break

            # --- IMPROVEMENT: Check if the window was closed with the 'X' button ---
            # cv2.getWindowProperty returns -1 when the window is closed by the user
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed by user, exiting.")
                    break
            except cv2.error:
                # This can happen if the window is closed while the property is being checked
                print("Window closed unexpectedly.")
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = make_args()
    # Renamed the main function to 'inference' to better reflect its purpose
    inference(args)