# Fruits and Vegetables Recognition using MobileNetV2

This project implements a deep learning model to recognize and classify 36 types of fruits and vegetables using computer vision and transfer learning techniques.

## 🧠 Problem Statement

Manual identification of fruits and vegetables is:

- Time-consuming  
- Prone to human error

By automating the recognition process, this system can improve workflows in:

- Agriculture  
- Retail checkout systems  
- Nutritional tracking  

## 📊 Dataset

- **Source:** Kaggle
- **Categories:** 36 types (e.g., Apple, Tomato, Carrot, Garlic, Mango, etc.)
- **Structure:**
  - Training: 100 images/class
  - Validation: 10 images/class
  - Testing: 10 images/class
- Images are organized in clearly labeled subfolders.

## 🧪 Methodology

- **Model Used:** MobileNetV2 (pretrained on ImageNet)
- **Modifications:**
  - `include_top=False` to remove the original classifier
  - Added:
    - Dense(128, activation='relu')
    - Dense(128, activation='relu')
    - Dense(36, activation='softmax')
- **Training:** Frozen base model; trained only top layers

## 🖼️ Data Preparation & Augmentation

- **Image Size:** 224×224 with 3 color channels
- **Preprocessing:** `mobilenet_v2.preprocess_input()`
- **Training Augmentations:**
  - Rotation (±30°)
  - Zoom (±15%)
  - Shear (15%)
  - Horizontal Flip
  - Width/Height Shift (±20%)
- **Validation/Test:** Resized and normalized without augmentation

## 📈 Model Performance

- **Training Accuracy:** Improved from 48% → ~93%
- **Validation Accuracy:** Improved from 88% → ~96%
- **Loss:** Smoothly decreasing, indicating no overfitting

## 🖼️ Static & Video Image Inference

- Preprocesses image: RGB conversion, normalization, resizing
- Loads model and predicts class label with confidence
- Accepts user-defined image paths

## 📷 Live Camera Inference

- Built using OpenCV
- Classifies fruits and vegetables from live webcam feed in real-time
- Displays predicted class and confidence on screen

## 🎥 Demo Highlights

- Augmentation improves generalization
- MobileNetV2 speeds up training and increases accuracy
- Consistent validation performance shows robustness

---

## 👨‍💻 Developed By

**Sarthik Ashokbhai Jagani**  
Student ID: 22306164  

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
