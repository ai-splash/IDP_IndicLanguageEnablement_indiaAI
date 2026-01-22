# IDP_IndicLanguageEnablement_indiaAI

# OCR Document Processing Pipeline

 An end-to-end OCR-centric Intelligent Document Processing (IDP) system designed for multilingual and heterogeneous Indian documents. The platform combines OCR, NLP, LLM-based extraction, and interactive visualization in a modular and scalable architecture.

<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/d47ab2c9-0f0f-4ca8-964a-1da7559619dc" />

## 🏗️ Architecture Overview

The system follows a modular, pipeline-based architecture:

1. Document ingestion and validation
2. OCR engine selection and routing
3. OCR text and layout extraction
4. Language detection and translation
5. AI-based entity extraction and summarization
6. MongoDB storage with audit trail
7. Web-based review and search interface

## ✨ Features

- **Multi-format Support**: PDF, JPG, JPEG, PNG, TIFF, BMP
- **Intelligent Processing**: Digital PDFs via Marker, scanned documents via Surya OCR
- **Multilingual Support**: Detects and translates Indian languages to English
- **Advanced Entity Extraction**: AI-powered extraction with Groq LLM
- **Interactive Visualization**: Hover-enabled bounding boxes with detailed metadata
- **MongoDB Storage**: Persistent storage of documents and extractions
- **Web Interface**: Multi-mode Streamlit GUI

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/BharatDBPG/India-AI-Intelligent-Document-Processing
   cd ocr
   ```

2. **Run setup script**:
   ```bash
   ./setup.sh
   ```

3. **Configure API keys**:
   ```bash
   # Update .streamlit/secrets.toml with your Groq API key
   GROQ_API_KEY = "your_key_here"
   ```

4. **Start the application**:
   ```bash
   ./start_app.sh
   ```

## 🖥️ Usage

Access the web interface at `http://localhost:8501` and use the four main tabs:
- **Upload & Process**: Upload and process documents
- **Analysis Results**: View extracted entities and translations
- **Document Viewer**: Interactive document visualization with bounding boxes
- **Search Documents**: Search across processed documents

## 📦 Requirements

- Python 3.8+
- MongoDB
- Groq API key

## 📄 License

MIT License

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.
