import os
from pymongo import MongoClient
from datetime import datetime

class DatabaseManager:
    def __init__(self, connection_string="mongodb://localhost:27017/", db_name="ocr_pipeline"):
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        
        # Existing collections
        self.documents = self.db.documents
        self.extractions = self.db.extractions
        self.logs = self.db.logs
        self.translations = self.db.translations  # Existing translation collection
        
        # NEW: Dedicated collection for processed results (used in search tab)
        self.processed_documents = self.db.processed_documents
   
    def store_document(self, filename, file_data, file_type):
        doc = {
            "filename": filename,
            "file_data": file_data,
            "file_type": file_type,
            "upload_time": datetime.now()
        }
        return self.documents.insert_one(doc).inserted_id
   
    def store_extraction(self, doc_id, extracted_data):
        extraction = {
            "document_id": doc_id,
            "extracted_fields": extracted_data.get("fields", {}),
            "confidence_scores": extracted_data.get("confidence", {}),
            "language": extracted_data.get("language", ""),
            "summary": extracted_data.get("summary", ""),
            "metadata": extracted_data.get("metadata", {})
        }
        return self.extractions.insert_one(extraction).inserted_id
   
    def store_translation(self, doc_id, translation_data):
        """Store translation and transliteration data"""
        translation = {
            "document_id": doc_id,
            "original_text": translation_data.get("original_text", ""),
            "detected_language": translation_data.get("detected_language", ""),
            "language_name": translation_data.get("language_name", ""),
            "translated_text": translation_data.get("translated_text", ""),
            "transliterated_text": translation_data.get("transliterated_text", "")
        }
        return self.translations.insert_one(translation).inserted_id
   
    def store_log(self, doc_id, extracted_text, page_data):
        log = {
            "document_id": doc_id,
            "extracted_text": extracted_text,
            "page_data": page_data
        }
        return self.logs.insert_one(log).inserted_id
   
    def get_extraction(self, doc_id):
        return self.extractions.find_one({"document_id": doc_id})
   
    def get_translation(self, doc_id):
        return self.translations.find_one({"document_id": doc_id})
    
    # ==================== NEW METHOD FOR SEARCH TAB ====================
    def store_processed_document(self, filename, raw_text, translated_text, entities, 
                                 summary="", document_type="", detected_language="", translation_stats=None):
        """
        Store fully processed document data for search and retrieval.
        Used by the Search tab to query across all documents.
        """
        searchable_text = (raw_text + " " + translated_text).lower()
        
        processed_data = {
            "original_filename": filename,
            "raw_text": raw_text,
            "translated_text": translated_text,
            "entities": entities,
            "summary": summary,
            "document_type": document_type,
            "detected_language": detected_language,
            "translation_stats": translation_stats or {},
            "searchable_text": searchable_text,  # For fast keyword search
            "processed_at": datetime.now()
        }
        
        # Upsert: update if filename exists, insert if new
        result = self.processed_documents.update_one(
            {"original_filename": filename},
            {"$set": processed_data},
            upsert=True
        )
        
        return result.upserted_id or result.matched_count > 0
    
    # Optional: Helper to get all processed documents (for debugging)
    def get_all_processed_documents(self):
        return list(self.processed_documents.find({}))