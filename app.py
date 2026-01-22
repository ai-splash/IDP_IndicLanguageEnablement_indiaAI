import streamlit as st
import tempfile
import os
import json
from PIL import Image, ImageDraw, ImageFont
import pymupdf
import io
import pandas as pd
from datetime import datetime
import sys
import re
import base64

# Add paths
sys.path.insert(0, '/home/boss/ocr/marker')
sys.path.insert(0, '/home/boss/ocr/surya')

from database import DatabaseManager
from deep_translator import GoogleTranslator

# Safe Groq import
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

# Safe Marker import
try:
    from marker.converters.pdf import PdfConverter
    from marker.config.parser import ConfigParser
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False

st.set_page_config(page_title="Advanced OCR Processor", layout="wide")

class DocumentViewer:
    """Document viewer with bounding box overlays and extracted content"""
    
    def __init__(self):
        self.font_size = 12
        
    def get_text_size(self, text, font):
        """Get text size for positioning"""
        im = Image.new(mode="P", size=(0, 0))
        draw = ImageDraw.Draw(im)
        _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
        return width, height
    
    def render_bboxes_on_image(self, image, bboxes, labels=None, colors=None):
        """Render bounding boxes and labels on image"""
        if not bboxes:
            return image
            
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.font_size)
        except:
            font = ImageFont.load_default()
        
        for i, bbox in enumerate(bboxes):
            # Ensure bbox coordinates are integers
            bbox = [int(p) for p in bbox]
            
            # Choose color
            color = colors[i] if colors and i < len(colors) else "red"
            
            # Draw bounding box
            draw.rectangle(bbox, outline=color, width=2)
            
            # Draw label if provided
            if labels and i < len(labels):
                label = str(labels[i])[:50]  # Truncate long labels
                text_pos = (bbox[0] + 2, bbox[1] + 2)
                
                # Get text size for background
                try:
                    text_size = self.get_text_size(label, font)
                    if text_size[0] > 0 and text_size[1] > 0:
                        # Draw background rectangle
                        bg_bbox = (
                            text_pos[0] - 1,
                            text_pos[1] - 1,
                            text_pos[0] + text_size[0] + 1,
                            text_pos[1] + text_size[1] + 1
                        )
                        draw.rectangle(bg_bbox, fill="white", outline=color)
                        
                        # Draw text
                        draw.text(text_pos, label, fill=color, font=font)
                except:
                    # Fallback without background
                    draw.text(text_pos, label, fill=color, font=font)
        
        return img_copy
    
    def image_to_base64(self, img):
        """Convert PIL image to base64 string"""
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def display_document_with_overlay(self, image, analysis_data):
        """Display document with interactive bounding box overlay and extraction table"""
        if not analysis_data:
            st.image(image, use_container_width=True)
            return
        
        # Extract bounding boxes and text from analysis data
        bboxes = []
        texts = []
        block_types = []
        
        # Handle different data formats
        if isinstance(analysis_data, dict):
            if 'text_blocks' in analysis_data:
                for block in analysis_data['text_blocks']:
                    if 'bbox' in block:
                        bboxes.append(block['bbox'])
                        texts.append(block.get('text', ''))
                        block_types.append(block.get('block_type', 'text'))
            elif 'bounding_boxes' in analysis_data:
                for bbox_data in analysis_data['bounding_boxes']:
                    bboxes.append(bbox_data['bbox'])
                    texts.append(bbox_data.get('text', ''))
                    block_types.append(bbox_data.get('block_type', 'text'))
        
        # Render overlay with numbered tags
        if bboxes:
            col1, col2 = st.columns([0.6, 0.4])
            
            with col1:
                overlay_image = self.render_numbered_bboxes(image, bboxes)
                st.image(overlay_image, use_container_width=True, caption="Document with Numbered Tags")
            
            with col2:
                st.subheader("📋 Extracted Text by Tag")
                
                # Create extraction table
                table_data = []
                for i, (bbox, text, block_type) in enumerate(zip(bboxes, texts, block_types)):
                    table_data.append({
                        'Tag': i + 1,
                        'Type': block_type.title(),
                        'Text': text[:100] + "..." if len(text) > 100 else text,
                        'Full Text': text
                    })
                
                # Display as dataframe
                if table_data:
                    df = pd.DataFrame(table_data)
                    st.dataframe(df[['Tag', 'Type', 'Text']], use_container_width=True)
                    
                    # Show full text on selection
                    selected_tag = st.selectbox("Select tag to view full text:", 
                                               options=[f"Tag {i+1}" for i in range(len(table_data))],
                                               key="tag_selector")
                    
                    if selected_tag:
                        tag_idx = int(selected_tag.split()[1]) - 1
                        st.text_area("Full Text:", table_data[tag_idx]['Full Text'], height=150)
        else:
            st.image(image, use_container_width=True)
            st.info("No text blocks detected for overlay")
    
    def render_numbered_bboxes(self, image, bboxes):
        """Render bounding boxes with numbered tags"""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        for i, bbox in enumerate(bboxes):
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                
                # Draw numbered tag
                tag_text = str(i + 1)
                tag_bbox = draw.textbbox((0, 0), tag_text, font=font)
                tag_width = tag_bbox[2] - tag_bbox[0]
                tag_height = tag_bbox[3] - tag_bbox[1]
                
                # Tag background
                tag_x = x1
                tag_y = y1 - tag_height - 4
                if tag_y < 0:
                    tag_y = y1
                
                draw.rectangle([tag_x, tag_y, tag_x + tag_width + 8, tag_y + tag_height + 4], 
                             fill="red", outline="red")
                
                # Tag text
                draw.text((tag_x + 4, tag_y + 2), tag_text, fill="white", font=font)
        
        return img_copy

class AdvancedOCRProcessor:
    def __init__(self, groq_api_key):
        self.db = DatabaseManager()
        self.groq_client = None
        self.viewer = DocumentViewer()
        if GROQ_AVAILABLE and groq_api_key != "dummy":
            try:
                self.groq_client = Groq(api_key=groq_api_key)
            except:
                self.groq_client = None
        
    def store_input_files(self, uploaded_files):
        """Store uploaded files in input_table"""
        file_ids = []
        for file in uploaded_files:
            file.seek(0)  # Reset file pointer
            file_data = {
                'filename': file.name,
                'file_data': file.read(),
                'file_type': file.name.split('.')[-1].lower(),
                'file_size': len(file.getvalue()),
                'upload_time': datetime.now(),
                'status': 'uploaded'
            }
            file.seek(0)  # Reset again for later use
            file_id = self.db.db['input_table'].insert_one(file_data).inserted_id
            file_ids.append(str(file_id))
        return file_ids
    def process_with_marker(self, file_path):
        """Process document with Marker and extract bounding boxes"""
        if not MARKER_AVAILABLE:
            return None
            
        try:
            # Load models
            model_dict = create_model_dict()
            
            # Configure parser
            config_options = {
                "output_format": "json",
                "force_ocr": False,
                "debug": True,
                "output_dir": "/tmp/marker_debug"
            }
            config_parser = ConfigParser(config_options)
            config_dict = config_parser.generate_config_dict()
            config_dict["pdftext_workers"] = 1
            
            # Create converter
            converter = PdfConverter(
                config=config_dict,
                artifact_dict=model_dict,
                processor_list=config_parser.get_processors(),
                renderer=config_parser.get_renderer(),
                llm_service=config_parser.get_llm_service(),
            )
            
            # Process document
            rendered = converter(file_path)
            text, ext, images = text_from_rendered(rendered)
            
            # Extract bounding boxes from rendered data
            analysis_data = self.extract_bboxes_from_marker(rendered)
            
            return {
                'success': True,
                'text': text,
                'analysis_data': analysis_data,
                'images': images
            }
            
        except Exception as e:
            st.error(f"Marker processing failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def extract_bboxes_from_marker(self, rendered):
        """Extract bounding boxes from Marker debug data"""
        try:
            bboxes_data = []
            
            # Look for debug folder with blocks.json
            debug_dir = "/tmp/marker_debug"
            if os.path.exists(debug_dir):
                debug_folders = [d for d in os.listdir(debug_dir) if os.path.isdir(os.path.join(debug_dir, d))]
                if debug_folders:
                    latest_folder = max(debug_folders, key=lambda x: os.path.getctime(os.path.join(debug_dir, x)))
                    blocks_file = os.path.join(debug_dir, latest_folder, "blocks.json")
                    
                    if os.path.exists(blocks_file):
                        with open(blocks_file, 'r', encoding='utf-8') as f:
                            blocks_data = json.load(f)
                        
                        # Extract text blocks with bounding boxes
                        for block in blocks_data:
                            if 'children' in block:
                                for child in block['children']:
                                    if child.get('block_type') in ['23', '21', '20', '11'] and 'bbox' in child.get('polygon', {}):
                                        bbox = child['polygon']['bbox']
                                        
                                        # Get text from structure
                                        text = ""
                                        if 'structure' in child and child['structure']:
                                            for struct in child['structure']:
                                                # Find corresponding text block
                                                for text_block in blocks_data:
                                                    if 'children' in text_block:
                                                        for text_child in text_block['children']:
                                                            if (text_child.get('block_id') == struct.get('block_id') and 
                                                                'text' in text_child):
                                                                text += text_child['text'] + " "
                                        
                                        block_type_map = {
                                            '23': 'paragraph',
                                            '21': 'header', 
                                            '20': 'image',
                                            '11': 'chart'
                                        }
                                        
                                        bboxes_data.append({
                                            'bbox': bbox,
                                            'text': text.strip() or child.get('block_description', ''),
                                            'block_type': block_type_map.get(child.get('block_type'), 'text')
                                        })
            
            return {'bounding_boxes': bboxes_data} if bboxes_data else None
            
        except Exception as e:
            st.warning(f"Could not extract bounding boxes: {str(e)}")
            return None
    
    def process_with_marker_surya(self, file_path, filename):
        """Process document with OCR"""
        try:
            # Use OCR processor for all files
            from ocr_processor import OCRProcessor
            ocr_processor = OCRProcessor()
            
            if file_path.lower().endswith('.pdf'):
                result = ocr_processor.process_digital_pdf(file_path)
            else:
                result = ocr_processor.process_scanned_document(file_path)
            
            if 'text' in result:
                # Ensure bboxes exist in result
                if 'bboxes' not in result:
                    result['bboxes'] = []
                
                return {
                    'success': True,
                    'raw_text': result['text'],
                    'extracted_text': result['text'],
                    'metadata': result.get('metadata', {}),
                    'format': result.get('format', 'unknown'),
                    'bboxes': result.get('bboxes', []),
                    'engine': result.get('engine', 'marker_surya')
                }
            else:
                return {'success': False, 'error': 'No text extracted'}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _analyze_section_type(self, text):
        """Analyze section type based on text content"""
        text_lower = text.lower().strip()
        
        if not text_lower:
            return "empty"
        elif any(keyword in text_lower for keyword in ['name', 'नाम', 'பெயர்']):
            return "personal_info"
        elif any(keyword in text_lower for keyword in ['aadhaar', 'आधार', 'ஆதார்']):
            return "identification"
        elif any(keyword in text_lower for keyword in ['address', 'पता', 'முகவரி']):
            return "address"
        elif any(keyword in text_lower for keyword in ['phone', 'mobile', 'फोन', 'தொலைபேசி']):
            return "contact"
        elif text_lower.isupper() and len(text_lower.split()) < 5:
            return "header"
        elif len(text_lower.split()) < 3:
            return "label"
        else:
            return "content"
    
    def analyze_with_groq(self, raw_text):
        """Analyze text with Groq LLM and extract entities"""
        if not self.groq_client:
            return {
                "entities": {},
                "summary": "No summary available",
                "language": "unknown",
                "document_type": "unknown"
            }

        prompt = f"""
Analyze the following text and extract structured information.
Return ONLY valid JSON in this exact format:

{{
    "entities": {{
        "names": [],
        "aadhaar_numbers": [],
        "pan_numbers": [],
        "phone_numbers": [],
        "addresses": [],
        "dates": [],
        "amounts": [],
        "organizations": []
    }},
    "summary": "",
    "language": "",
    "document_type": ""
}}

Text to analyze:
{raw_text[:3000]}
"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )

            content = response.choices[0].message.content
            
            # Clean potential markdown code blocks
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            return json.loads(content)

        except json.JSONDecodeError as e:
            return {
                "entities": {},
                "summary": f"JSON parsing failed: {str(e)}",
                "language": "unknown",
                "document_type": "unknown"
            }
        except Exception as e:
            return {
                "entities": {},
                "summary": f"Analysis failed: {str(e)}",
                "language": "unknown",
                "document_type": "unknown"
            }
    
    def is_english_text(self, text):
        """Check if text is primarily English"""
        if not text or len(text.strip()) == 0:
            return True
        
        # Count English characters vs non-English
        english_chars = sum(1 for c in text if ord(c) < 128)
        total_chars = len(text.replace(' ', '').replace('\n', ''))
        
        if total_chars == 0:
            return True
        
        # If more than 80% ASCII characters, consider it English
        return (english_chars / total_chars) > 0.80
    
    def detect_language(self, text):
        """Detect language of text"""
        try:
            from langdetect import detect
            return detect(text)
        except:
            # Fallback: check for specific Unicode ranges
            if any('\u0900' <= c <= '\u097F' for c in text):  # Devanagari (Hindi)
                return 'hi'
            elif any('\u0B80' <= c <= '\u0BFF' for c in text):  # Tamil
                return 'ta'
            elif any('\u0C00' <= c <= '\u0C7F' for c in text):  # Telugu
                return 'te'
            else:
                return 'en'
    
    def translate_line_by_line(self, text):
        """Translate text line by line, detecting language for each line"""
        lines = text.split('\n')
        translated_lines = []
        translation_stats = {
            'total_lines': len(lines),
            'translated_lines': 0,
            'english_lines': 0,
            'languages_detected': set()
        }
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                translated_lines.append('')
                continue
            
            # Check if line is already English
            if self.is_english_text(line):
                translated_lines.append(line)
                translation_stats['english_lines'] += 1
                translation_stats['languages_detected'].add('en')
                continue
            
            # Detect language and translate
            try:
                detected_lang = self.detect_language(line)
                translation_stats['languages_detected'].add(detected_lang)
                
                if detected_lang != 'en':
                    translator = GoogleTranslator(source=detected_lang, target='en')
                    translated = translator.translate(line)
                    translated_lines.append(translated)
                    translation_stats['translated_lines'] += 1
                else:
                    translated_lines.append(line)
                    translation_stats['english_lines'] += 1
                    
            except Exception as e:
                # If translation fails, keep original
                translated_lines.append(line)
        
        return {
            'translated_text': '\n'.join(translated_lines),
            'stats': {
                'total_lines': translation_stats['total_lines'],
                'translated_lines': translation_stats['translated_lines'],
                'english_lines': translation_stats['english_lines'],
                'languages_detected': list(translation_stats['languages_detected'])
            }
        }
    
    def translate_to_english(self, text):
        """Enhanced translation with line-by-line detection and translation"""
        try:
            if not text or len(text.strip()) == 0:
                return {
                    "original_text": text,
                    "translated_text": text,
                    "detected_language": "unknown",
                    "confidence": 0.0,
                    "translation_stats": {}
                }
            
            # Use line-by-line translation for mixed language documents
            translation_result = self.translate_line_by_line(text)
            
            return {
                "original_text": text,
                "translated_text": translation_result['translated_text'],
                "detected_language": "mixed",
                "confidence": 0.92,
                "translation_stats": translation_result['stats']
            }

        except Exception as e:
            return {
                "original_text": text,
                "translated_text": text,
                "detected_language": "unknown",
                "confidence": 0.0,
                "error": str(e),
                "translation_stats": {}
            }


def convert_pdf_to_images(pdf_path):
    """Convert PDF to images"""
    images = []
    try:
        doc = pymupdf.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        doc.close()
    except Exception as e:
        st.error(f"Error converting PDF to images: {str(e)}")
    return images


def draw_bounding_boxes_with_sections(image, bboxes):
    """Draw bounding boxes with section-based colors"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Section-based color mapping
    section_colors = {
        'personal_info': 'red',
        'identification': 'blue', 
        'address': 'green',
        'contact': 'orange',
        'header': 'purple',
        'label': 'cyan',
        'content': 'gray',
        'empty': 'lightgray'
    }
    
    for i, bbox in enumerate(bboxes):
        section = bbox.get('section', 'content')
        color = section_colors.get(section, 'gray')
        coords = bbox.get('bbox', [])
        
        if len(coords) >= 4:
            x1, y1, x2, y2 = coords[:4]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Add section label
            label = f"{section[:3].upper()}{i+1}"
            try:
                # Try to use a default font
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            draw.text((x1, max(0, y1-25)), label, fill=color, font=font)
    
    return img_copy


def cleanup_temp_file(file_path):
    """Clean up temporary files"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        st.warning(f"Could not delete temporary file: {str(e)}")


def main():
    st.title("🚀 Advanced OCR Document Processor")
    
    # Initialize processor
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") or "dummy"
    processor = AdvancedOCRProcessor(groq_api_key)
    
    # Create 3 tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "📊 Analysis Results", "📄 Document Viewer", "🔍 Search Documents"])
    
    with tab1:
        st.header("Upload & Process Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files to process",
            type=['pdf', 'jpg', 'jpeg', 'png', 'tiff'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"📁 Uploaded {len(uploaded_files)} files")
            
            # AUTO-STORE files immediately upon upload
            if 'stored_file_ids' not in st.session_state or st.session_state.get('last_uploaded_count') != len(uploaded_files):
                with st.spinner("📥 Auto-storing files in database..."):
                    file_ids = processor.store_input_files(uploaded_files)
                    st.session_state['stored_file_ids'] = file_ids
                    st.session_state['last_uploaded_count'] = len(uploaded_files)
                    st.success(f"✅ Auto-stored {len(file_ids)} files in database")
            
            # Process files button (only after auto-storage)
            if st.button("🚀 Process with OCR & Translation", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                temp_files = []  # Track temp files for cleanup
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name} with OCR engines...")
                    
                    # Save file temporarily
                    file.seek(0)  # Reset file pointer
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = tmp.name
                        temp_files.append(tmp_path)
                    
                    try:
                        # OCR Processing
                        ocr_result = processor.process_with_marker_surya(tmp_path, file.name)
                        
                        if ocr_result.get('success'):
                            raw_text = ocr_result['raw_text']
                            
                            # Translation (Line-by-line for mixed languages)
                            status_text.text(f"🌐 Translating {file.name} line-by-line...")
                            translation_result = processor.translate_to_english(raw_text)
                            
                            # Groq Analysis (on translated text)
                            status_text.text(f"🤖 Analyzing {file.name} with Groq LLM...")
                            groq_analysis = processor.analyze_with_groq(translation_result['translated_text'])
                            
                            results.append({
                                'filename': file.name,
                                'file_path': tmp_path,
                                'ocr_result': ocr_result,
                                'groq_analysis': groq_analysis,
                                'translation': translation_result,
                                'success': True
                            })
                                                        # === SAVE TO DATABASE FOR SEARCH TAB ===
                            processor.db.store_processed_document(
                                filename=file.name,
                                raw_text=raw_text,
                                translated_text=translation_result['translated_text'],
                                entities=groq_analysis.get('entities', {}),
                                summary=groq_analysis.get('summary', ''),
                                document_type=groq_analysis.get('document_type', ''),
                                detected_language=translation_result.get('detected_language', ''),
                                translation_stats=translation_result.get('translation_stats', {})
                            )
                        else:
                            results.append({
                                'filename': file.name,
                                'file_path': tmp_path,
                                'error': ocr_result.get('error'),
                                'success': False
                            })
                    
                    except Exception as e:
                        results.append({
                            'filename': file.name,
                            'file_path': tmp_path,
                            'error': str(e),
                            'success': False
                        })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                st.session_state['processing_results'] = results
                st.session_state['temp_files'] = temp_files
                status_text.text("✅ All documents processed!")
    
    with tab2:
        st.header("📊 Analysis Results")
        
        if 'processing_results' in st.session_state:
            results = st.session_state['processing_results']
            
            # File selector
            filenames = [r['filename'] for r in results]
            selected_file = st.selectbox("Select document:", filenames)
            
            if selected_file:
                result_data = next(r for r in results if r['filename'] == selected_file)
                
                if result_data.get('success'):
                    groq_analysis = result_data['groq_analysis']
                    translation = result_data['translation']
                    
                    # Translation Statistics
                    st.subheader("🌐 Translation Statistics")
                    trans_stats = translation.get('translation_stats', {})
                    if trans_stats:
                        cols = st.columns(4)
                        cols[0].metric("Total Lines", trans_stats.get('total_lines', 0))
                        cols[1].metric("Translated", trans_stats.get('translated_lines', 0))
                        cols[2].metric("English Lines", trans_stats.get('english_lines', 0))
                        langs = trans_stats.get('languages_detected', [])
                        cols[3].metric("Languages", len(langs))
                        
                        if langs:
                            st.info(f"📝 Detected Languages: {', '.join(langs)}")
                    
                    # Entities Table
                    st.subheader("📋 Extracted Entities")
                    entities = groq_analysis.get('entities', {})
                    
                    table_data = []
                    for field, values in entities.items():
                        if values:
                            for value in values:
                                table_data.append({
                                    'Field': field.replace('_', ' ').title(),
                                    'Value': str(value),
                                    'Confidence Score': '0.95'
                                })
                    
                    if table_data:
                        df = pd.DataFrame(table_data)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No entities extracted")
                    
                    # Additional Info
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📝 Summary")
                        st.write(groq_analysis.get('summary', 'No summary available'))
                        
                        st.subheader("📄 Document Info")
                        st.write(f"**Type:** {groq_analysis.get('document_type', 'Unknown')}")
                        st.write(f"**Language:** {translation.get('detected_language', 'Unknown')}")
                    
                    with col2:
                        st.subheader("🔄 Translated Text")
                        translated_text = translation.get('translated_text', 'No translation available')
                        st.text_area("English Translation", translated_text, height=300, disabled=True)
                    
                    # Side by side comparison
                    st.subheader("🔀 Original vs Translated Comparison")
                    comp_col1, comp_col2 = st.columns(2)
                    with comp_col1:
                        st.text_area("Original Text", translation.get('original_text', '')[:2000], height=200, disabled=True)
                    with comp_col2:
                        st.text_area("Translated Text", translated_text[:2000], height=200, disabled=True)
                    
                    # Raw JSON
                    with st.expander("🔍 Raw JSON Data"):
                        st.json({
                            "groq_analysis": groq_analysis,
                            "translation": translation,
                            "ocr_engine": result_data['ocr_result'].get('engine')
                        })
                
                else:
                    st.error(f"Processing failed: {result_data.get('error')}")
        else:
            st.info("No results available. Process documents first.")
    
    with tab3:
        st.header("📄 Enhanced Document Viewer with Bounding Boxes")
        
        if 'processing_results' in st.session_state:
            results = st.session_state['processing_results']
            
            # File selector
            filenames = [r['filename'] for r in results]
            selected_file = st.selectbox("Select document for detailed view:", filenames, key="viewer_select")
            
            if selected_file:
                result_data = next(r for r in results if r['filename'] == selected_file)
                file_path = result_data['file_path']
                
                if os.path.exists(file_path) and result_data.get('success'):
                    st.subheader("📖 Document Analysis with Numbered Tags")
                    
                    # Processing options
                    use_marker = st.checkbox("🔬 Use Enhanced Marker Analysis", value=True, help="Use Marker for better bounding box detection")
                    
                    if use_marker and MARKER_AVAILABLE:
                        # Process with Marker
                        with st.spinner("Processing with Marker..."):
                            marker_result = processor.process_with_marker(file_path)
                            
                        if marker_result and marker_result.get('success'):
                            # Use debug images from Marker if available
                            debug_images = []
                            debug_dir = "/tmp/marker_debug"
                            
                            # Find the latest debug folder
                            if os.path.exists(debug_dir):
                                debug_folders = [d for d in os.listdir(debug_dir) if os.path.isdir(os.path.join(debug_dir, d))]
                                if debug_folders:
                                    latest_folder = max(debug_folders, key=lambda x: os.path.getctime(os.path.join(debug_dir, x)))
                                    debug_path = os.path.join(debug_dir, latest_folder)
                                    
                                    # Look for page images
                                    for i in range(10):  # Check up to 10 pages
                                        page_img_path = os.path.join(debug_path, f"pdf_page_{i}.png")
                                        if os.path.exists(page_img_path):
                                            debug_images.append(Image.open(page_img_path))
                                        else:
                                            break
                            
                            # Fallback to generated images if no debug images
                            if not debug_images:
                                if file_path.lower().endswith('.pdf'):
                                    debug_images = convert_pdf_to_images(file_path)
                                else:
                                    debug_images = [Image.open(file_path)]
                            
                            # Display with overlays
                            for i, img in enumerate(debug_images):
                                st.write(f"**Page {i+1}**")
                                processor.viewer.display_document_with_overlay(img, marker_result['analysis_data'])
                        else:
                            st.error("Marker processing failed. Falling back to basic OCR.")
                            use_marker = False
                    
                    if not use_marker or not MARKER_AVAILABLE:
                        # Fallback to basic OCR results
                        if file_path.lower().endswith('.pdf'):
                            images = convert_pdf_to_images(file_path)
                        else:
                            images = [Image.open(file_path)]
                        
                        # Get existing bounding boxes
                        bboxes = result_data['ocr_result'].get('bboxes', [])
                        
                        for i, img in enumerate(images):
                            st.write(f"**Page {i+1}**")
                            page_bboxes = [bbox for bbox in bboxes if bbox.get('page', 0) == i]
                            
                            # Convert to analysis format
                            analysis_data = {
                                'text_blocks': [
                                    {
                                        'bbox': bbox.get('bbox', []),
                                        'text': bbox.get('text', ''),
                                        'confidence': bbox.get('confidence', 0)
                                    }
                                    for bbox in page_bboxes
                                ]
                            }
                            
                            processor.viewer.display_document_with_overlay(img, analysis_data)
                
                else:
                    st.error("File not found or processing failed")
        else:
            st.info("No documents available. Process documents first.")
    with tab4:
        st.header("🔍 Search Across Processed Documents")

        search_query = st.text_input("Enter keyword(s) to search in raw or translated text:", placeholder="e.g. Aadhaar, John, Chennai")

        if search_query:
            with st.spinner("Searching documents..."):
                # Case-insensitive keyword search in combined text
                query_regex = re.compile(re.escape(search_query.strip()), re.IGNORECASE)
                cursor = processor.db.db['processed_documents'].find({
                    "searchable_text": {"$regex": query_regex}
                })

                documents = list(cursor)

                if documents:
                    st.success(f"Found {len(documents)} matching document(s)")

                    for doc in documents:
                        with st.expander(f"📄 {doc['original_filename']} • Processed on {doc['processed_at'].strftime('%Y-%m-%d %H:%M')}"):
                            col1, col2 = st.columns(2)

                            with col1:
                                st.subheader("📝 Raw Text (First 1000 chars)")
                                st.text_area("Raw", doc['raw_text'][:1000], height=200, disabled=True, key=f"raw_{doc['_id']}")

                                st.subheader("🌐 Translated Text (First 1000 chars)")
                                st.text_area("Translated", doc['translated_text'][:1000], height=200, disabled=True, key=f"trans_{doc['_id']}")

                            with col2:
                                st.subheader("📋 Extracted Entities")
                                entities = doc.get('entities', {})
                                entity_list = []
                                for field, values in entities.items():
                                    if values:
                                        for val in values:
                                            entity_list.append({"Field": field.replace('_', ' ').title(), "Value": val})
                                if entity_list:
                                    st.dataframe(pd.DataFrame(entity_list), use_container_width=True)
                                else:
                                    st.info("No entities found")

                                st.subheader("📄 Document Info")
                                st.write(f"**Type:** {doc.get('document_type', 'Unknown')}")
                                st.write(f"**Language:** {doc.get('detected_language', 'Unknown')}")
                                st.write(f"**Summary:** {doc.get('summary', 'No summary')}")

                else:
                    st.warning("No documents found matching your search.")
        else:
            st.info("Enter a keyword to search across all processed documents (searches both original and translated text).")

if __name__ == "__main__":
    main()
