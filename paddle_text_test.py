# [
#     [
#         [[[304.0, 59.0], [557.0, 59.0], [557.0, 164.0], [304.0, 164.0]], ('PORK', 0.9959486126899719)], 
#         [[[46.0, 90.0], [233.0, 75.0], [236.0, 113.0], [48.0, 128.0]], ('PREMIUM', 0.9945175051689148)], 
#         [[[57.0, 180.0], [808.0, 175.0], [809.0, 308.0], [58.0, 314.0]], ('LUNCHEON MEAT', 0.9788351655006409)]
#     ]
# ]
# result[0] -> returns entire list of detected text
# result[0][0] -> returns the first detected text
# result[0][0][0] -> returns the coordinates of the detected text
# result[0][0][1] -> returns the text detected and the confidence score


import cv2
import numpy as np
from paddleocr import PaddleOCR

# Load the image
def load_image(image_path):
    image = cv2.imread(image_path)
    return image

# Preprocess the image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# Detect and extract text using PaddleOCR
def extract_text(image_path, use_preprocessed=False):
    ocr = PaddleOCR()
    if use_preprocessed:
        image = preprocess_image(cv2.imread(image_path))
        temp_path = "temp_processed_image.jpg"
        cv2.imwrite(temp_path, image)
        image_path = temp_path
    
    result = ocr.ocr(image_path, cls=True)
    extracted_text = []
    
    for line in result:
        for word_info in line:
            text = word_info[1][0]  # Extract detected text
            extracted_text.append(text)
    
    return extracted_text

# Main function
def main(image_path):
    print("Running OCR on raw image...")
    raw_text = extract_text(image_path, use_preprocessed=False)
    print("Extracted Text (Raw Image):", raw_text)
    
    print("\nRunning OCR on preprocessed image...")
    preprocessed_text = extract_text(image_path, use_preprocessed=True)
    print("Extracted Text (Preprocessed Image):", preprocessed_text)

if __name__ == "__main__":
    main("/home/rse/prog/Capstone/Dataset/Visual/VP001/test_a/1.jpg")
