#!/usr/bin/env python3
"""
extract.py - Extract highlights from PDFs and images
"""

import os
import sys
import re
import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pathlib import Path

# Define directories
PDF_DIR = Path(__file__).parent / "pdfs"
IMAGE_DIR = Path(__file__).parent / "images"
OUTPUT_DIR = Path(__file__).parent / "output"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_highlights_from_pdf(pdf_path):
    """
    Extract highlighted text from a PDF file
    """
    print(f"Processing PDF: {pdf_path}")
    highlights = []
    
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # Iterate through each page
        for page_num, page in enumerate(doc):
            # Get annotations (including highlights)
            annots = page.annots()
            page_highlights = []
            if annots:
                for annot in annots:
                    # Check if it's a highlight annotation
                    if annot.type[0] == 8:  # 8 is the type for highlight
                        # Get the highlighted text
                        rect = annot.rect
                        highlight_text = page.get_text("text", clip=rect).strip()
                        if highlight_text:
                            page_highlights.append({
                                "text": highlight_text,
                                "page": page_num + 1,
                                "source": os.path.basename(pdf_path),
                                "y_pos": rect.y0,  # Store y-position for sorting
                                "y_end": rect.y1,  # Store end y-position for connecting highlights
                                "rect": rect       # Store the rectangle for later analysis
                            })
            
            # Alternative method: look for yellow/highlighted areas
            # Get page pixmap (image)
            pix = page.get_pixmap()
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            # Convert to HSV for better color detection
            if pix.n >= 3:  # If it has at least 3 channels (RGB)
                img_hsv = cv2.cvtColor(img[:,:,:3], cv2.COLOR_RGB2HSV)
                
                # Yellow highlight mask (adjust these values as needed)
                lower_yellow = np.array([20, 100, 100])
                upper_yellow = np.array([40, 255, 255])
                mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
                
                # Find contours of highlighted areas
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    if cv2.contourArea(contour) > 100:  # Filter out small areas
                        x, y, w, h = cv2.boundingRect(contour)
                        # Convert back to PDF coordinates
                        rect = fitz.Rect(x, y, x+w, y+h)
                        text = page.get_text("text", clip=rect).strip()
                        if text and len(text) > 3:  # Ignore very short texts
                            page_highlights.append({
                                "text": text,
                                "page": page_num + 1,
                                "source": os.path.basename(pdf_path),
                                "y_pos": y,       # Store y-position for sorting
                                "y_end": y + h,   # Store end y-position for connecting highlights
                                "rect": rect      # Store the rectangle for later analysis
                            })
            
            # Sort highlights by y-position (top to bottom)
            page_highlights.sort(key=lambda h: h["y_pos"])
            
            # Connect highlights that are likely part of the same paragraph
            connected_highlights = []
            current_highlight = None
            
            for h in page_highlights:
                if current_highlight is None:
                    current_highlight = h
                else:
                    # Check if this highlight is close to the previous one
                    # and if they form a coherent sentence when combined
                    y_gap = h["y_pos"] - current_highlight["y_end"]
                    
                    # Threshold for vertical distance (adjust as needed)
                    # This determines how close highlights need to be to be considered part of the same paragraph
                    threshold = 20  # Pixels or points
                    
                    # Check if the highlights are close enough vertically
                    if y_gap <= threshold:
                        # Check if they form a coherent sentence
                        # Handle hyphenation: if the first highlight ends with a hyphen, remove it
                        first_text = current_highlight["text"]
                        second_text = h["text"]
                        
                        # Check for hyphenation at the end of the first text
                        if first_text.endswith('-'):
                            # Remove the hyphen
                            first_text = first_text[:-1]
                            
                            # If the second text starts with a lowercase letter, it's likely a continuation of a word
                            if second_text and second_text[0].islower():
                                # Connect without space
                                combined_text = first_text + second_text
                            else:
                                # Connect with space
                                combined_text = first_text + " " + second_text
                        else:
                            # Normal connection with space
                            combined_text = first_text + " " + second_text
                        
                        # Simple heuristic: if the last word of the first highlight and the first word
                        # of the second highlight form a complete sentence, they're likely connected
                        words1 = current_highlight["text"].split()
                        words2 = h["text"].split()
                        
                        if len(words1) > 0 and len(words2) > 0:
                            last_char = current_highlight["text"][-1]
                            
                            # If the first highlight ends with a period, question mark, or exclamation mark,
                            # it's likely the end of a sentence
                            if last_char in ['.', '?', '!'] and not first_text.endswith('-'):
                                # Start a new highlight
                                connected_highlights.append(current_highlight)
                                current_highlight = h
                            else:
                                # Connect the highlights
                                current_highlight["text"] = combined_text
                                current_highlight["y_end"] = h["y_end"]
                                current_highlight["rect"] = fitz.Rect(
                                    min(current_highlight["rect"].x0, h["rect"].x0),
                                    current_highlight["rect"].y0,
                                    max(current_highlight["rect"].x1, h["rect"].x1),
                                    h["rect"].y1
                                )
                        else:
                            connected_highlights.append(current_highlight)
                            current_highlight = h
                    else:
                        connected_highlights.append(current_highlight)
                        current_highlight = h
            
            # Add the last highlight if there is one
            if current_highlight is not None:
                connected_highlights.append(current_highlight)
            
            # Remove the temporary fields before adding to the final list
            for h in connected_highlights:
                h.pop("y_pos", None)
                h.pop("y_end", None)
                h.pop("rect", None)
                highlights.append(h)
    
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
    
    return highlights

def extract_highlights_from_image(image_path):
    """
    Extract highlighted text from an image
    """
    print(f"Processing image: {image_path}")
    highlights = []
    page_highlights = []
    
    try:
        # Read the image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Could not read image: {image_path}")
            return highlights
            
        # Convert to HSV for better color detection
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common highlight colors
        color_ranges = {
            "yellow": ([20, 100, 100], [40, 255, 255]),
            "green": ([40, 40, 40], [80, 255, 255]),
            "blue": ([90, 40, 40], [130, 255, 255]),
            "pink": ([140, 40, 40], [170, 255, 255])
        }
        
        for color_name, (lower, upper) in color_ranges.items():
            lower_color = np.array(lower)
            upper_color = np.array(upper)
            
            # Create mask for this color
            mask = cv2.inRange(img_hsv, lower_color, upper_color)
            
            # Find contours of highlighted areas
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 200:  # Filter out small areas
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Add some padding to ensure we get the full text
                    x = max(0, x - 5)
                    y = max(0, y - 5)
                    w = min(img.shape[1] - x, w + 10)
                    h = min(img.shape[0] - y, h + 10)
                    
                    # Extract the region of interest
                    roi = img[y:y+h, x:x+w]
                    
                    # Use Tesseract to extract text from this region
                    text = pytesseract.image_to_string(roi).strip()
                    
                    if text and len(text) > 3:  # Ignore very short texts
                        page_highlights.append({
                            "text": text,
                            "color": color_name,
                            "source": os.path.basename(image_path),
                            "y_pos": y,       # Store y-position for sorting
                            "y_end": y + h,   # Store end y-position for connecting highlights
                            "bbox": (x, y, w, h)  # Store bounding box for later analysis
                        })
        
        # If no highlights found with color detection, try OCR on the whole image
        if not page_highlights:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get black and white image
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Use Tesseract to extract all text
            text = pytesseract.image_to_string(thresh).strip()
            
            # Split by lines and filter out empty lines
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            for i, line in enumerate(lines):
                if len(line) > 10:  # Only consider substantial lines
                    page_highlights.append({
                        "text": line,
                        "source": os.path.basename(image_path),
                        "note": "Extracted without color detection",
                        "y_pos": i,      # Use line number as position
                        "y_end": i + 1,  # End position is just below start
                        "bbox": (0, i, 100, 1)  # Dummy bounding box
                    })
        
        # Sort highlights by y-position (top to bottom)
        page_highlights.sort(key=lambda h: h["y_pos"])
        
        # Connect highlights that are likely part of the same paragraph
        connected_highlights = []
        current_highlight = None
        
        for h in page_highlights:
            if current_highlight is None:
                current_highlight = h
            else:
                # Check if this highlight is close to the previous one
                y_gap = h["y_pos"] - current_highlight["y_end"]
                
                # Threshold for vertical distance (adjust as needed)
                threshold = 20  # Pixels
                
                # Check if the highlights are close enough vertically
                if y_gap <= threshold:
                    # Check if they form a coherent sentence
                    # Handle hyphenation: if the first highlight ends with a hyphen, remove it
                    first_text = current_highlight["text"]
                    second_text = h["text"]
                    
                    # Check for hyphenation at the end of the first text
                    if first_text.endswith('-'):
                        # Remove the hyphen
                        first_text = first_text[:-1]
                        
                        # If the second text starts with a lowercase letter, it's likely a continuation of a word
                        if second_text and second_text[0].islower():
                            # Connect without space
                            combined_text = first_text + second_text
                        else:
                            # Connect with space
                            combined_text = first_text + " " + second_text
                    else:
                        # Normal connection with space
                        combined_text = first_text + " " + second_text
                    
                    # Simple heuristic: if the last word of the first highlight and the first word
                    # of the second highlight form a complete sentence, they're likely connected
                    words1 = current_highlight["text"].split()
                    words2 = h["text"].split()
                    
                    if len(words1) > 0 and len(words2) > 0:
                        last_char = current_highlight["text"][-1]
                        
                        # If the first highlight ends with a period, question mark, or exclamation mark,
                        # it's likely the end of a sentence
                        if last_char in ['.', '?', '!'] and not first_text.endswith('-'):
                            # Start a new highlight
                            connected_highlights.append(current_highlight)
                            current_highlight = h
                        else:
                            # Connect the highlights
                            current_highlight["text"] = combined_text
                            current_highlight["y_end"] = h["y_end"]
                            
                            # Update bounding box
                            x1, y1, w1, h1 = current_highlight["bbox"]
                            x2, y2, w2, h2 = h["bbox"]
                            current_highlight["bbox"] = (
                                min(x1, x2),
                                y1,
                                max(x1 + w1, x2 + w2) - min(x1, x2),
                                h["y_end"] - y1
                            )
                    else:
                        connected_highlights.append(current_highlight)
                        current_highlight = h
                else:
                    connected_highlights.append(current_highlight)
                    current_highlight = h
        
        # Add the last highlight if there is one
        if current_highlight is not None:
            connected_highlights.append(current_highlight)
        
        # Remove the temporary fields before adding to the final list
        for h in connected_highlights:
            h.pop("y_pos", None)
            h.pop("y_end", None)
            h.pop("bbox", None)
            highlights.append(h)
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
    
    return highlights

def save_highlights(highlights, output_file):
    """
    Save extracted highlights to a text file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Extracted Highlights\n\n")
        f.write(f"Total highlights found: {len(highlights)}\n\n")
        
        # Group highlights by source
        sources = {}
        for h in highlights:
            source = h.get('source', 'Unknown')
            if source not in sources:
                sources[source] = []
            sources[source].append(h)
        
        # Write highlights grouped by source
        for source, source_highlights in sources.items():
            f.write(f"## Source: {source}\n\n")
            
            for h in source_highlights:
                f.write(f"- {h['text']}\n")
                if 'page' in h:
                    f.write(f"  (Page {h['page']})\n")
                if 'color' in h:
                    f.write(f"  (Highlight color: {h['color']})\n")
                f.write("\n")
            
            f.write("\n")

def main():
    all_highlights = []
    
    # Process PDFs
    if PDF_DIR.exists():
        for pdf_file in PDF_DIR.glob("*.pdf"):
            highlights = extract_highlights_from_pdf(pdf_file)
            all_highlights.extend(highlights)
            print(f"Found {len(highlights)} highlights in {pdf_file.name}")
    else:
        print(f"PDF directory not found: {PDF_DIR}")
    
    # Process images
    if IMAGE_DIR.exists():
        for img_file in IMAGE_DIR.glob("*.jpg"):
            highlights = extract_highlights_from_image(img_file)
            all_highlights.extend(highlights)
            print(f"Found {len(highlights)} highlights in {img_file.name}")
        
        # Also check for png files
        for img_file in IMAGE_DIR.glob("*.png"):
            highlights = extract_highlights_from_image(img_file)
            all_highlights.extend(highlights)
            print(f"Found {len(highlights)} highlights in {img_file.name}")
    else:
        print(f"Image directory not found: {IMAGE_DIR}")
    
    # Save all highlights
    if all_highlights:
        output_file = OUTPUT_DIR / "highlights.md"
        save_highlights(all_highlights, output_file)
        print(f"\nExtracted {len(all_highlights)} total highlights")
        print(f"Results saved to {output_file}")
    else:
        print("No highlights found")

if __name__ == "__main__":
    main()
