import os
import tempfile
from typing import List, Optional, BinaryIO
from uuid import UUID
import hashlib
from sqlalchemy.orm import Session
from minio import Minio
from minio.error import S3Error
from fastapi import UploadFile
from app.models import Document, Folder
from app.config import settings
from app.core.exceptions import NotFoundException, BadRequestException
from app.utils import (
    get_file_type,
    is_supported_file_type,
    extract_text_from_file,
    validate_file_size
)
# Import MCP components for CSV/Excel handling
from app.mcp.data_processor import MCPDataProcessor
from app.config import settings

class DocumentService:
    def __init__(self, db: Session):
        self.db = db
        self.minio_client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure
        )
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Ensure the MinIO bucket exists"""
        try:
            if not self.minio_client.bucket_exists(settings.minio_bucket):
                self.minio_client.make_bucket(settings.minio_bucket)
        except S3Error as e:
            print(f"Error creating bucket: {e}")
    
    def _generate_file_hash(self, file_content: bytes) -> str:
        """Generate SHA-256 hash of file content"""
        return hashlib.sha256(file_content).hexdigest()
    
    def _get_object_name(self, document_id: str, filename: str) -> str:
        """Generate object name for MinIO storage"""
        return f"documents/{document_id}/{filename}"
    
    async def upload_document(
        self,
        file: UploadFile,
        folder_id: UUID,
        uploaded_by: UUID
    ) -> Document:
        """Upload a document to MinIO and save metadata to database"""
        # Validate folder exists
        folder = self.db.query(Folder).filter(Folder.id == folder_id).first()
        if not folder:
            raise NotFoundException("Folder not found")
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Validate file size
        if not validate_file_size(file_size):
            raise BadRequestException("File size exceeds maximum limit (50MB)")
        
        # Get file type
        file_type = get_file_type(file.filename)
        if not file_type:
            raise BadRequestException("Could not determine file type")
        
        # Generate file hash for deduplication
        file_hash = self._generate_file_hash(file_content)
        
        # Check if file already exists in folder
        existing_doc = self.db.query(Document).filter(
            Document.folder_id == folder_id,
            Document.filename == file.filename
        ).first()
        
        if existing_doc:
            raise BadRequestException("File with this name already exists in the folder")
        
        # Create document record
        document = Document(
            folder_id=folder_id,
            filename=file.filename,
            file_type=file_type,
            file_size=file_size,
            file_path="",  # Will be updated after MinIO upload
            doc_metadata={"file_hash": file_hash},
            uploaded_by=uploaded_by
        )
        
        self.db.add(document)
        self.db.flush()  # Get the document ID
        
        # Upload to MinIO
        object_name = self._get_object_name(str(document.id), file.filename)

        try:
            # Create a temporary file for upload
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file.seek(0)
                temp_file_path = temp_file.name

            try:
                # Now use the temp file for MinIO upload
                self.minio_client.fput_object(
                    settings.minio_bucket,
                    object_name,
                    temp_file_path,
                    content_type=file.content_type
                )
            finally:
                # Clean up temp file with a small delay to avoid Windows file lock issues
                import time
                time.sleep(0.1)  # Give MinIO time to finish reading
                try:
                    os.unlink(temp_file_path)
                except Exception as cleanup_error:
                    print(f"Warning: Failed to clean up temp file {temp_file_path}: {cleanup_error}")
            
            # For CSV/Excel files, also upload to MCP
            mcp_file_id = None
            if file_type.lower() in ['csv', 'xlsx', 'xls']:
                try:
                    print(f"Attempting MCP upload for {file.filename}")
                    mcp_processor = MCPDataProcessor(settings, self.db)
                    mcp_result = await mcp_processor.upload_file(
                        file_data=file_content,
                        filename=file.filename,
                        folder_id=str(folder_id),
                        user_id=str(uploaded_by)
                    )
                    mcp_file_id = mcp_result.get('file_id')
                    print(f"MCP upload result: {mcp_result}")
                    if mcp_file_id:
                        print(f"MCP file_id set: {mcp_file_id}")
                    else:
                        print("MCP upload failed - no file_id returned")
                except Exception as e:
                    # Log error but don't fail document upload
                    print(f"Warning: Failed to upload to MCP: {str(e)}")
                    import traceback
                    traceback.print_exc()

            # Update document with file path BEFORE embeddings (need it for MinIO access)
            document.file_path = object_name

            # Update document metadata with MCP info
            if mcp_file_id:
                existing_metadata = document.doc_metadata or {}
                updated_metadata = {**existing_metadata, 'mcp_file_id': mcp_file_id}
                document.doc_metadata = updated_metadata
                print(f"Updated document metadata: {updated_metadata}")

            # For unstructured documents (PDFs, DOCs, etc.), generate embeddings immediately
            # AFTER file_path is set but BEFORE commit so session is still valid
            if file_type.lower() not in ['csv', 'xlsx', 'xls']:
                try:
                    print(f"Processing embeddings for {file.filename} (type: {file_type})")
                    from app.services.embedding_service import EmbeddingService
                    embedding_service = EmbeddingService(self.db)

                    # Process document embeddings
                    print(f"Starting embedding process for document {document.id}")
                    embeddings = await embedding_service.process_document_embeddings(
                        document_id=document.id,
                        chunk_size=500,  # Good balance of context and searchability
                        overlap=100
                    )
                    print(f"Successfully generated {len(embeddings)} embedding chunks for {file.filename}")

                except Exception as e:
                    # Log error but don't fail document upload
                    print(f"Warning: Failed to generate embeddings for {file.filename}: {str(e)}")
                    import traceback
                    traceback.print_exc()

            # Now commit the document with all metadata
            self.db.commit()
            # Don't refresh after commit - document is already saved

            return document
            
        except S3Error as e:
            self.db.rollback()
            raise BadRequestException(f"Failed to upload file: {str(e)}")
    
    def get_document(self, document_id: UUID) -> Optional[Document]:
        """Get document by ID"""
        return self.db.query(Document).filter(Document.id == document_id).first()
    
    def get_documents_in_folder(self, folder_id: UUID) -> List[Document]:
        """Get all documents in a folder"""
        return self.db.query(Document).filter(Document.folder_id == folder_id).all()
    
    def download_document(self, document_id: UUID) -> tuple[BinaryIO, str, str]:
        """Download document from MinIO"""
        document = self.get_document(document_id)
        if not document:
            raise NotFoundException("Document not found")
        
        try:
            response = self.minio_client.get_object(
                settings.minio_bucket,
                document.file_path
            )
            return response, document.filename, document.file_type
            
        except S3Error as e:
            raise BadRequestException(f"Failed to download file: {str(e)}")
    
    def delete_document(self, document_id: UUID) -> bool:
        """Delete document from both database and MinIO"""
        document = self.get_document(document_id)
        if not document:
            raise NotFoundException("Document not found")
        
        try:
            # Delete from MinIO
            self.minio_client.remove_object(
                settings.minio_bucket,
                document.file_path
            )
            
            # Delete from database (this will cascade to embeddings)
            self.db.delete(document)
            self.db.commit()
            
            return True
            
        except S3Error as e:
            self.db.rollback()
            raise BadRequestException(f"Failed to delete file: {str(e)}")
    
    def extract_document_text(self, document_id: UUID) -> str:
        """Extract text content from document"""
        document = self.get_document(document_id)
        if not document:
            raise NotFoundException("Document not found")
        
        if not is_supported_file_type(document.file_type):
            raise BadRequestException(f"File type '{document.file_type}' is not supported for text extraction")
        
        try:
            # Download file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{document.file_type}") as temp_file:
                response = self.minio_client.get_object(
                    settings.minio_bucket,
                    document.file_path
                )

                # Write content to temp file
                for chunk in response.stream(32*1024):
                    temp_file.write(chunk)
                temp_file.flush()
                temp_file_path = temp_file.name
                temp_file.close()  # Close explicitly before extraction

            try:
                # Extract text from the closed temp file
                text = extract_text_from_file(temp_file_path, document.file_type)
                return text
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file_path)
                except Exception as cleanup_error:
                    print(f"Warning: Failed to clean up temp file {temp_file_path}: {cleanup_error}")
                
        except S3Error as e:
            raise BadRequestException(f"Failed to download file for text extraction: {str(e)}")
        except Exception as e:
            raise BadRequestException(f"Failed to extract text: {str(e)}")
    
    def update_document_metadata(
        self,
        document_id: UUID,
        metadata: dict
    ) -> Document:
        """Update document metadata"""
        document = self.get_document(document_id)
        if not document:
            raise NotFoundException("Document not found")

        # Merge with existing metadata
        existing_metadata = document.doc_metadata or {}
        existing_metadata.update(metadata)
        document.doc_metadata = existing_metadata

        self.db.commit()
        self.db.refresh(document)

        return document

    async def create_document_from_file(
        self,
        folder_id: UUID,
        file_path: str,
        filename: str,
        file_size: int,
        uploaded_by: UUID,
        content_type: str = None
    ) -> Document:
        """
        Create a document from a file path (used for provider sync).

        Args:
            folder_id: Target folder UUID
            file_path: Path to local file
            filename: Original filename
            file_size: File size in bytes
            uploaded_by: User UUID
            content_type: Optional MIME type

        Returns:
            Created Document instance

        Raises:
            NotFoundException: If folder not found
            BadRequestException: If file validation or upload fails
        """
        # Validate folder exists
        folder = self.db.query(Folder).filter(Folder.id == folder_id).first()
        if not folder:
            raise NotFoundException("Folder not found")

        # Validate file size
        if not validate_file_size(file_size):
            raise BadRequestException("File size exceeds maximum limit (50MB)")

        # Get file type
        file_type = get_file_type(filename)
        if not file_type:
            raise BadRequestException("Could not determine file type")

        # Read file and generate hash
        with open(file_path, "rb") as f:
            file_content = f.read()
        file_hash = self._generate_file_hash(file_content)

        # Check if file already exists in folder
        existing_doc = self.db.query(Document).filter(
            Document.folder_id == folder_id,
            Document.filename == filename
        ).first()

        if existing_doc:
            # Update filename to avoid conflict (append timestamp)
            import time
            base, ext = os.path.splitext(filename)
            filename = f"{base}_{int(time.time())}{ext}"

        # Create document record
        document = Document(
            folder_id=folder_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            file_path="",  # Will be updated after MinIO upload
            doc_metadata={"file_hash": file_hash, "source": "provider_sync"},
            uploaded_by=uploaded_by
        )

        self.db.add(document)
        self.db.flush()  # Get the document ID

        # Upload to MinIO
        object_name = self._get_object_name(str(document.id), filename)

        try:
            # Determine content type if not provided
            if not content_type:
                content_type = f"application/{file_type}"

            self.minio_client.fput_object(
                settings.minio_bucket,
                object_name,
                file_path,
                content_type=content_type
            )

            # Update document with file path
            document.file_path = object_name
            self.db.commit()
            self.db.refresh(document)

            return document

        except S3Error as e:
            self.db.rollback()
            raise BadRequestException(f"Failed to upload file to storage: {str(e)}")
