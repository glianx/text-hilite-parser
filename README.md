# PDF and Image Highlight Extractor

This tool extracts highlighted text from PDF files and images.

## Requirements

- Python 3.6+
- Tesseract OCR engine (for image processing)

## Installation

1. Install Tesseract OCR:
   - On macOS: `brew install tesseract`
   - On Ubuntu: `sudo apt-get install tesseract-ocr`
   - On Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Place your PDF files in the `pdfs` directory
2. Place your image files (JPG, PNG) in the `images` directory
3. Run the script:
   ```
   python extract.py
   ```
4. The extracted highlights will be saved to `output/highlights.md`

## How It Works

- For PDFs: Extracts text from highlight annotations and also detects highlighted areas based on color
- For Images: Uses color detection to find highlighted regions and OCR to extract text from those regions

## Notes

- The color detection is optimized for yellow highlights but also attempts to detect green, blue, and pink highlights
- If no highlights are detected in an image, the script will attempt to extract all text from the image
- Adjust the color ranges in the script if your highlights are not being detected correctly
