import sys
import os
sys.path.append('/home/boss/ocr/surya')

from PIL import Image
# from image_preprocessor import ImagePreprocessor  # Moved to not_used
import json

class OCRProcessor:
    def __init__(self):
        # self.preprocessor = ImagePreprocessor()  # Removed - not needed
        pass
        
    def process_digital_pdf(self, pdf_path):
        """Process digital PDF using Surya OCR (treat as scanned)"""
        return self.process_scanned_document(pdf_path)
    
    def process_scanned_document(self, file_path):
        """Process any document using Surya OCR"""
        try:
            # Import surya modules
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor
            from surya.foundation import FoundationPredictor
            from surya.common.surya.schema import TaskNames
            
            # Load models
            foundation_predictor = FoundationPredictor()
            det_predictor = DetectionPredictor()
            rec_predictor = RecognitionPredictor(foundation_predictor)
            
            # Load and preprocess image/PDF
            if file_path.lower().endswith('.pdf'):
                import pymupdf
                doc = pymupdf.open(file_path)
                images = []
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    preprocessed_img = img  # Use image directly
                    images.append(preprocessed_img)
                doc.close()
            else:
                original_image = Image.open(file_path)
                images = [original_image]
            
            # Run OCR
            task_names = [TaskNames.ocr_with_boxes] * len(images)
            predictions = rec_predictor(
                images,
                task_names=task_names,
                det_predictor=det_predictor,
                math_mode=False
            )
            
            # Format output with layout analysis
            # Format output with layout analysis
            result = {
                "text": "",
                "metadata": {
                    "pages": len(images),
                    "layout_analysis": {
                        "sections": [],
                        "tables": [],
                        "headers": [],
                        "page_structure": []
                    }
                },
                "format": "json"
            }
            
            for i, pred in enumerate(predictions):
                page_text = ""
                page_data = []
                bboxes_for_nav = []
                
                for text_line in pred.text_lines:
                    page_text += text_line.text + "\n"
                    bbox_data = {
                        "text": text_line.text,
                        "bbox": text_line.bbox,
                        "confidence": getattr(text_line, 'confidence', 0.9)
                    }
                    page_data.append(bbox_data)
                    
                    # Add to navigation metadata
                    bboxes_for_nav.append({
                        "type": "text_line",
                        "page": i + 1,
                        "bbox": text_line.bbox,
                        "text_preview": text_line.text[:50]
                    })
                
                result["text"] += f"## Page {i+1}\n{page_text}\n"
                result["metadata"]["layout_analysis"]["page_structure"].append({
                    "page_number": i+1,
                    "text_lines": page_data,
                    "bboxes": bboxes_for_nav
                })
            
            return result
            
        except Exception as e:
            print(f"Error processing document with Surya: {e}")
            return None
