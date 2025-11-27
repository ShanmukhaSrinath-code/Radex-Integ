"""
Data Processor for MCP Module

Handles CSV/Excel file processing, caching, and metadata tracking within RADEX.
Uses MinIO for storage instead of local filesystem.
"""

import os
import time
import pandas as pd
import warnings
from typing import Dict, List, Any, Optional
from functools import lru_cache
from datetime import datetime

from minio import Minio
from minio.error import S3Error
from ..core.exceptions import BadRequestException


class MCPDataProcessor:
    """Handles data processing for MCP analysis within RADEX"""

    def __init__(self, settings, db_session=None):
        self.settings = settings
        self.db = db_session  # Database session for persistence
        self.minio_client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure
        )
        # Ensure bucket exists
        try:
            if not self.minio_client.bucket_exists(settings.minio_bucket):
                self.minio_client.make_bucket(settings.minio_bucket)
        except S3Error as e:
            print(f"Warning: Could not create MinIO bucket: {e}")
        self.df_cache = {}  # Cache for DataFrames
        self.CACHE_TIMEOUT = 300  # 5 minutes
        self.file_metadata = {}  # Keep local cache for performance, but persist to DB

    async def upload_file(self, file_data: bytes, filename: str, folder_id: str, user_id: str) -> Dict[str, Any]:
        """Upload a file to MinIO and validate it, with database persistence"""
        from app.models import McpFileMetadata

        try:
            # Validate file type
            if not filename.lower().endswith(('.csv', '.xlsx', '.xls')):
                raise BadRequestException(f"Unsupported file type: {filename}. Only CSV and Excel files are allowed.")

            # Upload to MinIO with MCP subdirectory
            object_name = f"mcp/{user_id}/{folder_id}/{filename}"
            # Convert bytes to file for upload (following RADEX pattern)
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file_data)
                temp_file.flush()
                temp_file_path = temp_file.name

            try:
                try:
                    self.minio_client.fput_object(
                        self.settings.minio_bucket,
                        object_name,
                        temp_file_path,
                        content_type="application/octet-stream"
                    )
                except S3Error as e:
                    raise BadRequestException(f"Failed to upload file to storage: {str(e)}")
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file_path)
                except Exception as cleanup_error:
                    print(f"Warning: Failed to clean up temp file {temp_file_path}: {cleanup_error}")

            # Validate by reading the file
            df = await self._read_file_from_minio(object_name)

            file_id = f"{user_id}_{folder_id}_{filename}"

            # Analyze file content to understand intent and extract keywords
            content_analysis = self._analyze_file_content(df, filename)

            # Store enhanced metadata in database
            if self.db:
                db_metadata = McpFileMetadata(
                    file_id=file_id,
                    user_id=user_id,
                    folder_id=folder_id,
                    filename=filename,
                    object_name=object_name,
                    file_type='csv' if filename.lower().endswith('.csv') else 'excel',
                    file_size=len(file_data),
                    row_count=len(df),
                    columns=list(df.columns)
                )

                # Store enhanced metadata as JSON in doc_metadata field
                enhanced_metadata = {
                    'content_analysis': content_analysis,
                    'domain': content_analysis['domain'],
                    'intent_keywords': content_analysis['intent_keywords'],
                    'semantic_tags': content_analysis['semantic_tags']
                }
                db_metadata.doc_metadata = enhanced_metadata

                self.db.add(db_metadata)
                self.db.commit()
                print(f"Stored enhanced MCP file metadata in database: {file_id}")
                print(f"  Domain: {content_analysis['domain']}, Keywords: {content_analysis['intent_keywords'][:3]}")
            else:
                # Fallback to in-memory if no DB
                metadata = {
                    'upload_time': time.time(),
                    'file_size': len(file_data),
                    'file_type': 'csv' if filename.lower().endswith('.csv') else 'excel',
                    'columns': list(df.columns),
                    'row_count': len(df),
                    'folder_id': folder_id,
                    'user_id': user_id,
                    'object_name': object_name,
                    'filename': filename,
                    'content_analysis': content_analysis
                }
                self.file_metadata[file_id] = metadata
                print(f"Stored enhanced MCP file metadata in memory: {file_id}")

            return {
                'file_id': file_id,
                'filename': filename,
                'size': len(file_data),
                'columns': list(df.columns),
                'row_count': len(df),
                'upload_time': datetime.now()
            }

        except Exception as e:
            # Clean up failed upload
            try:
                self.minio_client.remove_object(self.settings.minio_bucket, object_name)
            except:
                pass  # Ignore cleanup errors
            raise BadRequestException(f"File upload failed: {str(e)}")

    async def read_file(self, file_id: str) -> pd.DataFrame:
        """Read and cache DataFrame from MinIO"""
        current_time = time.time()

        # Check cache first
        if file_id in self.df_cache:
            cached_df, timestamp = self.df_cache[file_id]
            if current_time - timestamp < self.CACHE_TIMEOUT:
                # Update access time
                if file_id in self.file_metadata:
                    self.file_metadata[file_id]['last_accessed'] = current_time
                return cached_df
            else:
                # Cache expired
                del self.df_cache[file_id]

        # Get metadata from cache or database
        if file_id not in self.file_metadata:
            # If we have DB access, try to query the database
            if self.db:
                try:
                    from app.models import McpFileMetadata
                    print(f"DEBUG: Querying database for file_id: {file_id}")
                    db_file = self.db.query(McpFileMetadata).filter(McpFileMetadata.file_id == file_id).first()
                    print(f"DEBUG: Database query result for file {file_id}: {'found' if db_file else 'not found'}")
                    if db_file:
                        # Load metadata into cache
                        metadata = {
                            'user_id': db_file.user_id,
                            'folder_id': db_file.folder_id,
                            'filename': db_file.filename,
                            'columns': db_file.columns,
                            'row_count': db_file.row_count,
                            'file_size': db_file.file_size,
                            'object_name': db_file.object_name,
                            'upload_time': db_file.upload_time.timestamp() if hasattr(db_file.upload_time, 'timestamp') else float(db_file.upload_time),
                            'last_accessed': db_file.last_accessed.timestamp() if db_file.last_accessed and hasattr(db_file.last_accessed, 'timestamp') else None
                        }
                        self.file_metadata[file_id] = metadata
                        print(f"Loaded file metadata from database for file_id: {file_id}")
                        print(f"DEBUG: Loaded metadata object_name: {metadata['object_name']}")
                    else:
                        # File not in MCP database, try to create metadata on-the-fly
                        print(f"MCP metadata not found for file {file_id}, attempting on-the-fly metadata creation...")

                        # Extract components from file_id format: {user_id}_{folder_id}_{filename}
                        # User_id and folder_id are UUIDs (36 chars), filename is the rest
                        # Split into exactly 3 parts by taking first UUID, second UUID, and rest as filename
                        import re
                        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'

                        match = re.match(rf'^({uuid_pattern})_({uuid_pattern})_(.*)$', file_id)
                        if match:
                            user_id, folder_id, filename = match.groups()
                            print(f"Extracted user_id: {user_id}, folder_id: {folder_id}, filename: {filename}")

                            # Try to read the file directly from MinIO
                            object_name = f"mcp/{user_id}/{folder_id}/{filename}"
                            try:
                                df = await self._read_file_from_minio(object_name)
                                print(f"Successfully read file from MinIO, creating MCP metadata...")

                                # Analyze file content
                                content_analysis = self._analyze_file_content(df, filename)

                                # Create MCP metadata in database
                                if self.db:
                                    from uuid import uuid4
                                    db_metadata = McpFileMetadata(
                                        file_id=file_id,
                                        user_id=user_id,
                                        folder_id=folder_id,
                                        filename=filename,
                                        object_name=object_name,
                                        file_type='csv' if filename.lower().endswith('.csv') else 'excel',
                                        file_size=0,  # We don't know, but will be set later if needed
                                        row_count=len(df),
                                        columns=list(df.columns)
                                    )

                                    # Store enhanced metadata as JSON in doc_metadata field
                                    enhanced_metadata = {
                                        'content_analysis': content_analysis,
                                        'domain': content_analysis['domain'],
                                        'intent_keywords': content_analysis['intent_keywords'],
                                        'semantic_tags': content_analysis['semantic_tags']
                                    }
                                    db_metadata.doc_metadata = enhanced_metadata

                                    self.db.add(db_metadata)
                                    self.db.commit()
                                    print(f"Created MCP metadata in database for on-the-fly file: {file_id}")

                                    # Store in memory cache
                                    metadata = {
                                        'upload_time': current_time,
                                        'file_size': 0,
                                        'file_type': 'csv' if filename.lower().endswith('.csv') else 'excel',
                                        'columns': list(df.columns),
                                        'row_count': len(df),
                                        'folder_id': folder_id,
                                        'user_id': user_id,
                                        'object_name': object_name,
                                        'filename': filename,
                                        'content_analysis': content_analysis
                                    }
                                    self.file_metadata[file_id] = metadata
                                    print(f"Stored MCP file metadata in memory cache: {file_id}")
                                else:
                                    print("No database connection available for MCP metadata creation")
                                    raise BadRequestException(f"File metadata not found and no database for on-the-fly creation: {file_id}")
                            except Exception as minio_err:
                                print(f"Failed to read file from MinIO for metadata creation: {minio_err}")
                                raise BadRequestException(f"File not found in MCP database and couldn't read from storage: {file_id}")
                        else:
                            print(f"Couldn't parse file_id format: {file_id}")
                            raise BadRequestException(f"File metadata not found and couldn't parse file_id: {file_id}")

                        # Let's see all files in DB to debug
                        all_files = self.db.query(McpFileMetadata).all()
                        print(f"DEBUG: All file IDs in database: {[f.file_id for f in all_files]}")
                except Exception as db_err:
                    print(f"Database query failed for file {file_id}: {db_err}")
                    import traceback
                    traceback.print_exc()
                    raise BadRequestException(f"File metadata not found: {file_id}")
            else:
                raise BadRequestException(f"File metadata not found: {file_id}")

        metadata = self.file_metadata[file_id]
        object_name = metadata['object_name']

        df = await self._read_file_from_minio(object_name)

        # Cache the DataFrame
        self.df_cache[file_id] = (df.copy(), current_time)

        # Update metadata access time
        metadata['last_accessed'] = current_time

        return df

    async def _read_file_from_minio(self, object_name: str) -> pd.DataFrame:
        """Read file from MinIO and return DataFrame"""
        try:
            # Get object from MinIO
            response = self.minio_client.get_object(self.settings.minio_bucket, object_name)

            # Create temporary file for pandas to read
            temp_path = f"/tmp/{object_name.split('/')[-1]}"
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)

            with open(temp_path, 'wb') as f:
                # Read all data from the response stream
                for chunk in response.stream(32*1024):
                    f.write(chunk)

            try:
                # Read with pandas
                if object_name.lower().endswith('.csv'):
                    df = pd.read_csv(temp_path)
                elif object_name.lower().endswith(('.xlsx', '.xls')):
                    # Suppress openpyxl warnings about unsupported extensions
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        df = pd.read_excel(temp_path)
                else:
                    raise BadRequestException("Unsupported file format")

                return df

            finally:
                # Clean up temp file
                try:
                    os.remove(temp_path)
                except:
                    pass

        except Exception as e:
            raise BadRequestException(f"Error reading file from storage: {str(e)}")

    async def sync_with_database(self, db_session):
        """Sync in-memory cache with database metadata"""
        if not db_session:
            return

        try:
            from app.models import McpFileMetadata

            print("Syncing MCP file metadata with database...")
            db_files = db_session.query(McpFileMetadata).all()

            for db_file in db_files:
                metadata = {
                    'user_id': db_file.user_id,
                    'folder_id': db_file.folder_id,
                    'filename': db_file.filename,
                    'columns': db_file.columns,
                    'row_count': db_file.row_count,
                    'file_size': db_file.file_size,
                    'object_name': db_file.object_name,
                    'upload_time': db_file.upload_time.timestamp() if hasattr(db_file.upload_time, 'timestamp') else float(db_file.upload_time),
                    'last_accessed': db_file.last_accessed.timestamp() if db_file.last_accessed and hasattr(db_file.last_accessed, 'timestamp') else None
                }
                self.file_metadata[db_file.file_id] = metadata

            print(f"Synced {len(self.file_metadata)} MCP files from database")
        except Exception as e:
            print(f"Error syncing with database: {str(e)}")

    async def list_user_files(self, user_id: str, folder_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List files uploaded by a user, optionally filtered by folder - checks database and cache"""
        from app.models import McpFileMetadata

        files_info = []

        try:
            # First try to read from database if we have one
            if self.db:
                query = self.db.query(McpFileMetadata).filter(McpFileMetadata.user_id == user_id)
                if folder_id:
                    query = query.filter(McpFileMetadata.folder_id == folder_id)

                db_files = query.all()

                for db_file in db_files:
                    # Update local cache for performance
                    metadata = {
                        'user_id': db_file.user_id,
                        'folder_id': db_file.folder_id,
                        'filename': db_file.filename,
                        'columns': db_file.columns,
                        'row_count': db_file.row_count,
                        'file_size': db_file.file_size,
                        'object_name': db_file.object_name,
                        'upload_time': db_file.upload_time.timestamp() if hasattr(db_file.upload_time, 'timestamp') else float(db_file.upload_time),
                        'last_accessed': db_file.last_accessed.timestamp() if db_file.last_accessed and hasattr(db_file.last_accessed, 'timestamp') else None
                    }
                    self.file_metadata[db_file.file_id] = metadata

                    files_info.append({
                        "file_id": db_file.file_id,
                        "filename": db_file.filename,
                        "folder_id": db_file.folder_id,
                        "columns": db_file.columns,
                        "row_count": db_file.row_count,
                        "file_size": db_file.file_size,
                        "upload_time": metadata['upload_time'],
                        "last_accessed": metadata['last_accessed']
                    })

                print(f"Read {len(files_info)} MCP files from database for user {user_id}")
                return files_info

            else:
                # Check if we have any files in memory that might be database-synced from elsewhere
                print(f"No direct database access, checking cached files for user {user_id}")
                for file_id, metadata in self.file_metadata.items():
                    if metadata['user_id'] == user_id:
                        if folder_id is None or metadata.get('folder_id') == folder_id:
                            files_info.append({
                                "file_id": file_id,
                                "filename": metadata['filename'],
                                "columns": metadata['columns'],
                                "row_count": metadata['row_count'],
                                "file_size": metadata['file_size'],
                                "upload_time": metadata['upload_time'],
                                "last_accessed": metadata.get('last_accessed', metadata['upload_time'])
                            })

                if not files_info:
                    print(f"No files found for user {user_id}, attempting to sync from a new DB session")

                    # Try to create a temporary DB session to sync files
                    try:
                        from app.database import SessionLocal
                        temp_db = SessionLocal()
                        await self.sync_with_database(temp_db)
                        temp_db.close()

                        # Retry the search after syncing
                        for file_id, metadata in self.file_metadata.items():
                            if metadata['user_id'] == user_id:
                                if folder_id is None or metadata.get('folder_id') == folder_id:
                                    files_info.append({
                                        "file_id": file_id,
                                        "filename": metadata['filename'],
                                        "folder_id": metadata['folder_id'],
                                        "columns": metadata['columns'],
                                        "row_count": metadata['row_count'],
                                        "file_size": metadata['file_size'],
                                        "upload_time": metadata['upload_time'],
                                        "last_accessed": metadata.get('last_accessed', metadata['upload_time'])
                                    })

                        if files_info:
                            print(f"After sync, found {len(files_info)} MCP files from database for user {user_id}")
                        else:
                            print(f"Sync completed but still no files found for user {user_id}")

                    except Exception as sync_error:
                        print(f"Database sync failed: {str(sync_error)}")

                return files_info

        except Exception as e:
            print(f"Error listing user files: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    async def describe_file(self, file_id: str) -> Dict[str, Any]:
        """Get detailed information about a file"""
        df = await self.read_file(file_id)

        columns_info = []
        for col in df.columns:
            columns_info.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].count()),
                "null_count": int(df[col].isnull().sum())
            })

        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": columns_info,
            "memory_usage": df.memory_usage(deep=True).sum(),
            "dtypes_summary": df.dtypes.value_counts().to_dict()
        }

    async def get_columns(self, file_id: str) -> List[str]:
        """Get column names for a file"""
        df = await self.read_file(file_id)
        return list(df.columns)

    async def execute_query(self, file_id: str, operation: str, **kwargs) -> Any:
        """Execute a query operation on the data"""
        df = await self.read_file(file_id)

        # Apply filter if provided
        filter_expr = kwargs.get('filter')
        if filter_expr:
            try:
                df = df.query(filter_expr)
            except Exception as e:
                raise BadRequestException(f"Invalid filter expression: {str(e)}")

        # Execute operation
        if operation == "head":
            n = kwargs.get('n', 5)
            return df.head(n).to_dict('records')

        elif operation == "average" and 'column' in kwargs:
            column = kwargs['column']
            if column not in df.columns:
                raise BadRequestException(f"Column '{column}' not found")
            return float(df[column].mean())

        elif operation == "sum" and 'column' in kwargs:
            column = kwargs['column']
            if column not in df.columns:
                raise BadRequestException(f"Column '{column}' not found")
            return float(df[column].sum())

        elif operation == "execute":
            code = kwargs.get('code')
            if not code:
                raise BadRequestException("Code parameter required for execute operation")
            try:
                # Execute pandas code in a controlled environment
                result = eval(code, {"pd": pd, "df": df})
                if hasattr(result, 'to_dict'):
                    if isinstance(result, pd.DataFrame):
                        return result.to_dict('records')
                    else:
                        return result.to_dict()
                else:
                    return result
            except Exception as e:
                raise BadRequestException(f"Error executing code: {str(e)}")

        elif operation == "count":
            return len(df)

        elif operation == "describe":
            return df.describe().to_dict()

        elif operation == "groupby_count" and 'groupby_column' in kwargs:
            groupby_column = kwargs['groupby_column']
            if groupby_column not in df.columns:
                raise BadRequestException(f"Column '{groupby_column}' not found")

            # Apply filter if provided
            if filter_expr:
                try:
                    df = df.query(filter_expr)
                except Exception as e:
                    raise BadRequestException(f"Invalid filter expression: {str(e)}")

            # Group by and count
            result = df.groupby(groupby_column).size().reset_index(name='count')
            result = result.sort_values('count', ascending=False)

            # Limit results if requested
            limit = kwargs.get('limit', 10)
            result = result.head(limit)

            return result.to_dict('records')

        elif operation == "groupby_sum" and 'groupby_column' in kwargs and 'value_column' in kwargs:
            groupby_column = kwargs['groupby_column']
            value_column = kwargs['value_column']
            if groupby_column not in df.columns:
                raise BadRequestException(f"Column '{groupby_column}' not found")
            if value_column not in df.columns:
                raise BadRequestException(f"Column '{value_column}' not found")

            # Apply filter if provided
            if filter_expr:
                try:
                    df = df.query(filter_expr)
                except Exception as e:
                    raise BadRequestException(f"Invalid filter expression: {str(e)}")

            # Group by and sum
            result = df.groupby(groupby_column)[value_column].sum().reset_index()
            result = result.sort_values(value_column, ascending=False)

            # Limit results if requested
            limit = kwargs.get('limit', 10)
            result = result.head(limit)

            return result.to_dict('records')

        elif operation == "groupby_max" and 'groupby_column' in kwargs and 'value_column' in kwargs:
            groupby_column = kwargs['groupby_column']
            value_column = kwargs['value_column']
            if groupby_column not in df.columns:
                raise BadRequestException(f"Column '{groupby_column}' not found")
            if value_column not in df.columns:
                raise BadRequestException(f"Column '{value_column}' not found")

            # Apply filter if provided
            if filter_expr:
                try:
                    df = df.query(filter_expr)
                except Exception as e:
                    raise BadRequestException(f"Invalid filter expression: {str(e)}")

            # Group by and find max
            result = df.groupby(groupby_column)[value_column].max().reset_index()
            result = result.sort_values(value_column, ascending=False)

            # Limit results if requested
            limit = kwargs.get('limit', 10)
            result = result.head(limit)

            return result.to_dict('records')

        elif operation == "groupby_min" and 'groupby_column' in kwargs and 'value_column' in kwargs:
            groupby_column = kwargs['groupby_column']
            value_column = kwargs['value_column']
            if groupby_column not in df.columns:
                raise BadRequestException(f"Column '{groupby_column}' not found")
            if value_column not in df.columns:
                raise BadRequestException(f"Column '{value_column}' not found")

            # Apply filter if provided
            if filter_expr:
                try:
                    df = df.query(filter_expr)
                except Exception as e:
                    raise BadRequestException(f"Invalid filter expression: {str(e)}")

            # Group by and find min
            result = df.groupby(groupby_column)[value_column].min().reset_index()
            result = result.sort_values(value_column, ascending=True)

            # Limit results if requested
            limit = kwargs.get('limit', 10)
            result = result.head(limit)

            return result.to_dict('records')

        elif operation == "filter_count" and 'filter' in kwargs:
            # Count with filter (already handled by filter_expr above)
            filtered_count = len(df)
            return filtered_count

        elif operation == "unique_values" and 'column' in kwargs:
            column = kwargs['column']
            if column not in df.columns:
                raise BadRequestException(f"Column '{column}' not found")

            # Apply filter if provided
            if filter_expr:
                try:
                    df = df.query(filter_expr)
                except Exception as e:
                    raise BadRequestException(f"Invalid filter expression: {str(e)}")

            # Get unique values
            unique_vals = df[column].unique()
            return list(unique_vals)

        elif operation == "value_counts" and 'column' in kwargs:
            column = kwargs['column']
            if column not in df.columns:
                raise BadRequestException(f"Column '{column}' not found")

            # Apply filter if provided
            if filter_expr:
                try:
                    df = df.query(filter_expr)
                except Exception as e:
                    raise BadRequestException(f"Invalid filter expression: {str(e)}")

            # Get value counts
            result = df[column].value_counts().reset_index()
            result.columns = [column, 'count']
            result = result.sort_values('count', ascending=False)

            # Limit results if requested
            limit = kwargs.get('limit', 10)
            result = result.head(limit)

            return result.to_dict('records')

        else:
            raise BadRequestException(f"Unsupported operation: {operation}")

    def _analyze_file_content(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Analyze DataFrame content to understand its domain, intent, and semantic tags"""
        # Convert column names to lowercase for analysis
        columns_lower = [str(col).lower() for col in df.columns]
        filename_lower = filename.lower()

        # Analyze column patterns to determine domain
        domain_keywords = {
            'business': ['sales', 'revenue', 'profit', 'cost', 'price', 'amount', 'total', 'income', 'expense', 'margin'],
            'customer': ['customer', 'client', 'user', 'person', 'contact', 'email', 'phone', 'name', 'age', 'gender'],
            'product': ['product', 'item', 'sku', 'category', 'brand', 'model', 'type', 'color', 'size'],
            'geographic': ['country', 'region', 'state', 'city', 'zip', 'zipcode', 'postal', 'address', 'location'],
            'temporal': ['date', 'time', 'month', 'year', 'day', 'quarter', 'week', 'period', 'timestamp'],
            'financial': ['amount', 'balance', 'credit', 'debit', 'payment', 'transaction', 'account', 'currency'],
            'inventory': ['stock', 'quantity', 'inventory', 'supply', 'demand', 'units', 'available'],
            'sports': ['player', 'team', 'score', 'game', 'match', 'season', 'points', 'runs', 'bats'],
        }

        # Determine primary domain
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if any(keyword in col for col in columns_lower))
            if score > 0:
                domain_scores[domain] = score

        # Get primary domain (highest score)
        primary_domain = max(domain_scores, key=domain_scores.get) if domain_scores else 'general'

        # Extract intent keywords from column names
        all_keywords = set()
        for keywords in domain_keywords.values():
            all_keywords.update(keywords)

        # Filter columns that match domain keywords
        matching_keywords = []
        for col in columns_lower:
            for keyword in all_keywords:
                if keyword in col and keyword not in matching_keywords:
                    matching_keywords.append(keyword)

        # If no matches, use first few column names as generic keywords
        if not matching_keywords:
            matching_keywords = columns_lower[:3]

        # Generate semantic tags
        semantic_tags = [primary_domain]

        # Add more specific tags based on filename patterns
        if 'player' in filename_lower or 'stats' in filename_lower:
            semantic_tags.append('sports_statistics')
            semantic_tags.append('athletes')
        if 'store' in filename_lower or 'sales' in filename_lower:
            semantic_tags.append('retail_analytics')
        if 'superstore' in filename_lower:
            semantic_tags.append('supermarket_data')

        return {
            'domain': primary_domain,
            'intent_keywords': matching_keywords[:5],  # Limit to top 5
            'semantic_tags': semantic_tags,
            'column_count': len(df.columns),
            'row_count': len(df),
            'has_numeric': any(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns),
            'has_categorical': any(pd.api.types.is_object_dtype(df[col]) for col in df.columns),
            'analysis_timestamp': time.time()
        }

    async def delete_file(self, file_id: str) -> bool:
        """Delete a file and its metadata"""
        if file_id not in self.file_metadata:
            return False

        metadata = self.file_metadata[file_id]

        try:
            # Delete from MinIO
            self.minio_client.remove_object(self.settings.minio_bucket, metadata['object_name'])

            # Clean up cache and metadata
            if file_id in self.df_cache:
                del self.df_cache[file_id]
            del self.file_metadata[file_id]

            return True
        except Exception as e:
            print(f"Error deleting file {file_id}: {e}")
