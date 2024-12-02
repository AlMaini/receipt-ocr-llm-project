import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import os
from openai import OpenAI
import argparse

class ReceiptProcessor:
    def __init__(self):

        # Read OpenAI API key from file
        with open('openai-key.txt', 'r') as file:
            api_key = file.read().strip()
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        # Set Tesseract path for Windows users
        if os.name == 'nt':  # Windows
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def preprocess_image(self, image):
        """Apply image preprocessing steps to improve OCR accuracy"""
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply thresholding to get black and white image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Noise removal using median blur
        denoised = cv2.medianBlur(thresh, 3)
        
        return denoised

    def extract_text(self, image):
        """Extract text from image using Tesseract OCR"""
        # Get OCR data including bounding boxes
        data = pytesseract.image_to_data(image, output_type=Output.DICT)
        
        # Extract the full text
        text = pytesseract.image_to_string(image)
        
        return text, data

    def correct_with_llm(self, text):
        """Use OpenAI's GPT to correct and structure the receipt text"""
        prompt = f"""Please analyze this receipt text and return a structured JSON with:
        - Store name
        - Date
        - Items (with prices)
        - Total amount
        - Tax amount (if present)
        
        Clean up any OCR errors and format numbers properly.
        
        Receipt text:
        {text}"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes receipt text and returns structured data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content

    def visualize_boxes(self, image, data):
        """Draw bounding boxes around detected text"""
        boxes = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
        n_boxes = len(data['level'])
        
        for i in range(n_boxes):
            if int(data['conf'][i]) > 60:  # Only show confident detections
                (x, y, w, h) = (data['left'][i], data['top'][i], 
                               data['width'][i], data['height'][i])
                boxes = cv2.rectangle(boxes, (x, y), (x + w, y + h), 
                                    (0, 255, 0), 2)
        
        return boxes

    def process_receipt(self, image_path, show_visualization=False):
        """Main method to process a receipt image"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Preprocess
        processed = self.preprocess_image(image)
        
        # Extract text
        text, data = self.extract_text(processed)
        
        # Visualize if requested
        if show_visualization:
            boxes = self.visualize_boxes(processed, data)
            cv2.imshow('Detected Text', boxes)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Correct with LLM
        structured_data = self.correct_with_llm(text)
        
        return structured_data

def main():
    parser = argparse.ArgumentParser(description='Process a receipt image using OCR and LLM correction')
    parser.add_argument('image_path', help='Path to the receipt image')
    parser.add_argument('--visualize', action='store_true', help='Show visualization of detected text')
    args = parser.parse_args()

    processor = ReceiptProcessor()
    
    try:
        result = processor.process_receipt(args.image_path, args.visualize)
        print("\nProcessed Receipt Data:")
        print(result)
    except Exception as e:
        print(f"Error processing receipt: {str(e)}")

if __name__ == "__main__":
    main() 