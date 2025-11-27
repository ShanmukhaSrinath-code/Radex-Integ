import time
from typing import List, Dict, Any, Optional
from uuid import UUID
import openai
import json
from sqlalchemy.orm import Session
from app.models import User
from app.config import settings
from app.core.exceptions import BadRequestException, PermissionDeniedException
from app.services.permission_service import PermissionService
from app.services.embedding_service import EmbeddingService
from app.schemas import RAGQuery, RAGResponse, RAGChunk, ChatRequest, ChatResponse, ChatMessage
from app.models import Document
# Enhanced MCP integration for natural language CSV/Excel processing

import re

class RAGService:

    def _analyze_question_structure(self, question_lower: str) -> Dict[str, Any]:
        """
        Analyze question structure to detect query patterns like aggregation, filtering, etc.
        Returns query type and additional metadata.
        """
        # Pattern: "X wise Y" or "Y by X" (e.g., "region wise sales" = group sales by region)
        wise_pattern = re.search(r'\b(\w+)\s+wise\s+(\w+)', question_lower)
        by_pattern = re.search(r'\b(\w+)\s+by\s+(\w+)', question_lower)

        # Detect "wise" queries like "region wise sales"
        if wise_pattern:
            group_by_entity = wise_pattern.group(1)  # "region"
            measure_entity = wise_pattern.group(2)   # "sales"
            return {
                "query_type": "aggregation_wise",
                "key_terms": [group_by_entity, measure_entity, "wise"],
                "aggregation_info": {
                    "groupby_column": group_by_entity,
                    "measure_column": measure_entity,
                    "operation": "groupby"
                }
            }

        # Detect "by" queries like "sales by region"
        if by_pattern:
            measure_entity = by_pattern.group(1)     # "sales"
            group_by_entity = by_pattern.group(2)   # "region"
            return {
                "query_type": "aggregation_by",
                "key_terms": [measure_entity, group_by_entity, "by"],
                "aggregation_info": {
                    "groupby_column": group_by_entity,
                    "measure_column": measure_entity,
                    "operation": "groupby"
                }
            }

        # Detect multi-region queries like "east and west region sales"
        region_and_pattern = re.search(r'\b(\w+)\s+and\s+(\w+)\s+region\s+(\w+)', question_lower)
        if region_and_pattern:
            region1, region2, measure = region_and_pattern.groups()
            return {
                "query_type": "multi_region_filter",
                "key_terms": [region1, region2, measure, "region", "and"],
                "aggregation_info": {
                    "regions": [region1, region2],
                    "measure_column": measure,
                    "operation": "multi_filter_sum"
                }
            }

        # Detect single region queries like "east region sales"
        region_single_pattern = re.search(r'\b(\w+)\s+region\s+(\w+)', question_lower)
        if region_single_pattern:
            region, measure = region_single_pattern.groups()
            return {
                "query_type": "single_region_filter",
                "key_terms": [region, measure, "region"],
                "aggregation_info": {
                    "region": region,
                    "measure_column": measure,
                    "operation": "single_filter"
                }
            }

        # Detect simple filter queries like "california sales"
        filter_pattern = re.search(r'\b(\w+)\s+(\w+)', question_lower)
        if filter_pattern:
            entity1, entity2 = filter_pattern.groups()
            return {
                "query_type": "simple_filter",
                "key_terms": [entity1, entity2],
                "aggregation_info": {}
            }

        # Default case
        return {
            "query_type": "unknown",
            "key_terms": [],
            "aggregation_info": {}
        }

    def _analyze_dataframe_question_patterns(self, question_lower: str, columns: List[str]) -> Dict[str, Any]:
        """
        Analyzes question patterns to identify specific dataframe operations needed.
        Maps natural language to pandas operations for better code generation.
        """
        # Define column name patterns to look for
        column_patterns = {
            'regions': ['region', 'country/region', 'state', 'city'],
            'measures': ['sales', 'profit', 'quantity', 'price', 'amount', 'total', 'revenue'],
            'categories': ['category', 'product name', 'sub-category', 'department'],
            'entities': ['customer', 'customer name', 'segment', 'market']
        }

        # Find matching columns in the dataset
        found_columns = {
            'group_columns': [],
            'measure_columns': [],
            'filter_columns': []
        }

        lower_columns = [col.lower() for col in columns]

        for col_type, patterns in column_patterns.items():
            for pattern in patterns:
                for i, col in enumerate(lower_columns):
                    if pattern in col or col in pattern:
                        if col_type == 'regions':
                            found_columns['group_columns'].append(columns[i])
                        elif col_type == 'measures':
                            found_columns['measure_columns'].append(columns[i])
                        elif col_type == 'categories':
                            found_columns['group_columns'].append(columns[i])
                        elif col_type == 'entities':
                            found_columns['filter_columns'].append(columns[i])

        # Determine operation type based on question patterns
        operation_analysis = {
            'wisepattern': bool(re.search(r'\b(\w+)\s+wise\s+(\w+)', question_lower)),
            'bypattern': bool(re.search(r'\b(\w+)\s+by\s+(\w+)', question_lower)),
            'top_pattern': bool(re.search(r'\b(top|highest|most|max|maximum|best)\b', question_lower)),
            'bottom_pattern': bool(re.search(r'\b(bottom|lowest|least|min|minimum|worst)\b', question_lower)),
            'count_pattern': bool(re.search(r'\b(how many|count)\b', question_lower)),
            'average_pattern': bool(re.search(r'\b(average|mean|avg)\b', question_lower)),
            'sum_pattern': bool(re.search(r'\b(total|sum|amount)\b', question_lower))
        }

        # Determine primary operation
        primary_operation = "unknown"
        if operation_analysis['wisepattern']:
            primary_operation = "groupby_aggregate"
        elif operation_analysis['top_pattern'] and found_columns['measure_columns']:
            primary_operation = "top_n_groupby"
        elif operation_analysis['average_pattern']:
            primary_operation = "aggregate_mean"
        elif operation_analysis['sum_pattern']:
            primary_operation = "aggregate_sum"
        elif operation_analysis['count_pattern']:
            primary_operation = "count_rows"
        else:
            primary_operation = "inspect_data"

        return {
            'analysis': f"'{question_lower}' -> detected {sum(operation_analysis.values())} patterns",
            'primary_column': found_columns['measure_columns'][0] if found_columns['measure_columns'] else "unknown",
            'group_column': found_columns['group_columns'][0] if found_columns['group_columns'] else "unknown",
            'operation_type': primary_operation,
            'filters': [],
            'confidence': sum(operation_analysis.values()) / len(operation_analysis),
            'found_columns': found_columns,
            'detected_patterns': operation_analysis
        }
    def __init__(self, db: Session):
        self.db = db
        self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
        self.permission_service = PermissionService(db)
        self.embedding_service = EmbeddingService(db)
    
    async def query(
        self,
        user_id: UUID,
        rag_query: RAGQuery
    ) -> RAGResponse:
        """Process a RAG query and return response with sources"""
        start_time = time.time()
        
        try:
            # Get accessible folders for the user
            accessible_folders = self._get_accessible_folders(user_id, rag_query.folder_ids)
            
            if not accessible_folders:
                raise PermissionDeniedException("No accessible folders found for query")
            
            # Generate query embedding
            query_embedding = self.embedding_service.generate_embeddings([rag_query.query])[0]
            
            # Search for similar chunks
            similar_chunks = self.embedding_service.search_similar_chunks(
                query_embedding=query_embedding,
                folder_ids=accessible_folders,
                limit=rag_query.limit,
                min_similarity=rag_query.min_relevance_score
            )
            
            if not similar_chunks:
                return RAGResponse(
                    query=rag_query.query,
                    answer="No relevant documents found for your query.",
                    sources=[],
                    total_chunks=0,
                    processing_time=time.time() - start_time
                )
            
            # Generate answer using OpenAI
            answer = await self._generate_answer(rag_query.query, similar_chunks)
            
            # Format sources
            sources = []
            for chunk in similar_chunks:
                source = RAGChunk(
                    document_id=chunk["document_id"],
                    document_name=chunk["document_name"],
                    folder_id=chunk["folder_id"],
                    folder_name=chunk["folder_name"],
                    chunk_text=chunk["chunk_text"],
                    relevance_score=chunk["similarity_score"],
                    metadata=chunk["metadata"]
                )
                sources.append(source)
            
            processing_time = time.time() - start_time
            
            return RAGResponse(
                query=rag_query.query,
                answer=answer,
                sources=sources,
                total_chunks=len(similar_chunks),
                processing_time=processing_time
            )
            
        except Exception as e:
            if isinstance(e, (BadRequestException, PermissionDeniedException)):
                raise
            raise BadRequestException(f"Failed to process RAG query: {str(e)}")
    
    def _get_accessible_folders(
        self,
        user_id: UUID,
        requested_folder_ids: Optional[List[UUID]] = None
    ) -> List[UUID]:
        """Get list of folder IDs that user can access"""
        # Get all accessible folders for user (debug version)
        accessible_folders = self.permission_service.get_user_accessible_folders(user_id)
        accessible_folder_ids = []

        print(f"DEBUG: User {user_id} has access to {len(accessible_folders)} folders:")
        for folder in accessible_folders:
            accessible_folder_ids.append(folder.id)
            try:
                folder_name = getattr(folder, 'name', 'Unknown')
                folder_owner = getattr(folder, 'user_id', 'Unknown')
                print(f"  - Folder ID: {folder.id}, Name: {folder_name} (owner: {folder_owner})")
            except Exception as e:
                print(f"  - Folder ID: {folder.id} (error getting details: {e})")

        print(f"DEBUG: Accessible folder IDs: {accessible_folder_ids}")

        # If specific folders were requested, filter to only include accessible ones
        if requested_folder_ids:
            filtered_folder_ids = []
            for folder_id in requested_folder_ids:
                if folder_id in accessible_folder_ids:
                    filtered_folder_ids.append(folder_id)
                else:
                    print(f"DEBUG: Requested folder {folder_id} not in accessible list")
            return filtered_folder_ids

        return accessible_folder_ids
    
    async def _generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]]
    ) -> str:
        """Generate answer using OpenAI with provided context"""
        # Prepare context from chunks
        context_texts = []
        for chunk in context_chunks:
            context_texts.append(
                f"Document: {chunk['document_name']}\n"
                f"Content: {chunk['chunk_text']}\n"
                f"Relevance: {chunk['similarity_score']:.2f}\n"
            )
        
        context = "\n---\n".join(context_texts)
        
        # Create prompt
        system_prompt = """You are a helpful AI assistant that answers questions based on provided documents. 
Use only the information from the provided context to answer questions. 
If the context doesn't contain enough information to answer the question, say so clearly.
Cite the relevant documents when possible."""
        
        user_prompt = f"""Based on the following context documents, please answer this question: {query}

Context:
{context}

Answer:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.openai_chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            raise BadRequestException(f"Failed to generate answer: {str(e)}")
    
    def get_queryable_folders(self, user_id: UUID) -> List[Dict[str, Any]]:
        """Get list of folders that user can query"""
        accessible_folders = self.permission_service.get_user_accessible_folders(user_id)
        
        result = []
        for folder in accessible_folders:
            # Count documents in folder
            from app.models import Document
            document_count = self.db.query(Document).filter(
                Document.folder_id == folder.id
            ).count()
            
            # Count embeddings in folder
            from app.models import Embedding
            embedding_count = self.db.query(Embedding).join(Document).filter(
                Document.folder_id == folder.id
            ).count()
            
            result.append({
                "id": folder.id,
                "name": folder.name,
                "path": folder.path,
                "document_count": document_count,
                "embedding_count": embedding_count,
                "can_query": embedding_count > 0
            })
        
        return result
    
    async def suggest_related_queries(
        self,
        user_id: UUID,
        original_query: str,
        folder_ids: Optional[List[UUID]] = None
    ) -> List[str]:
        """Suggest related queries based on available content"""
        try:
            accessible_folders = self._get_accessible_folders(user_id, folder_ids)
            
            if not accessible_folders:
                return []
            
            # Get a sample of document titles and chunk texts for context
            from app.models import Document, Embedding
            
            # Get recent documents in accessible folders
            recent_docs = self.db.query(Document).filter(
                Document.folder_id.in_(accessible_folders)
            ).limit(10).all()
            
            doc_titles = [doc.filename for doc in recent_docs]
            
            # Create prompt for suggesting related queries
            system_prompt = """You are a helpful assistant that suggests related questions based on available documents.
Generate 3-5 related questions that someone might ask about the given documents."""
            
            user_prompt = f"""Based on these available documents: {', '.join(doc_titles)}
And the original query: "{original_query}"

Suggest 3-5 related questions that someone might ask:"""
            
            response = self.openai_client.chat.completions.create(
                model=settings.openai_chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0.8
            )
            
            suggestions_text = response.choices[0].message.content.strip()
            
            # Parse suggestions (assuming they're in a numbered list)
            suggestions = []
            for line in suggestions_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering and clean up
                    suggestion = line.split('.', 1)[-1].strip()
                    suggestion = suggestion.lstrip('- ').strip()
                    if suggestion:
                        suggestions.append(suggestion)
            
            return suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            # Return empty list on error rather than failing
            return []

    async def _reformulate_query(
        self,
        messages: List[ChatMessage],
        max_history: int = 5
    ) -> str:
        """
        Reformulate the latest user query based on conversation history.
        Takes last N messages for context and generates a standalone query.
        Falls back to original query if reformulation fails.
        """
        # Extract the latest user message
        user_messages = [msg for msg in messages if msg.role == "user"]
        if not user_messages:
            raise BadRequestException("No user messages found in conversation history")

        latest_query = user_messages[-1].content

        # If this is the first message or only one message, no reformulation needed
        if len(messages) <= 1:
            return latest_query

        # Take last N messages for context (excluding the latest)
        context_messages = messages[-(max_history + 1):-1] if len(messages) > max_history else messages[:-1]

        try:
            # Build context from conversation history
            conversation_context = "\n".join([
                f"{msg.role.upper()}: {msg.content}"
                for msg in context_messages
            ])

            system_prompt = """You are a query reformulation assistant. Your task is to reformulate user queries into standalone, self-contained questions based on conversation history.

Given a conversation history and the latest user query, reformulate the query to be completely standalone and contextually complete. The reformulated query should:
1. Include all necessary context from the conversation
2. Be understandable without reading the conversation history
3. Preserve the user's intent
4. Be suitable for semantic search over documents

Only return the reformulated query, nothing else."""

            user_prompt = f"""Conversation history:
{conversation_context}

Latest user query: {latest_query}

Reformulated standalone query:"""

            response = self.openai_client.chat.completions.create(
                model=settings.openai_reformulation_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0.3  # Lower temperature for more consistent reformulation
            )

            reformulated = response.choices[0].message.content.strip()

            # Validate reformulation - if it's empty, too short, or identical to original, fall back
            if not reformulated or len(reformulated) < 5 or reformulated.lower() == latest_query.lower():
                return latest_query

            return reformulated

        except Exception as e:
            # Fall back to original query on any error
            print(f"Query reformulation failed: {str(e)}. Using original query.")
            return latest_query

    async def chat(
        self,
        user_id: UUID,
        chat_request: ChatRequest
    ) -> ChatResponse:
        """
        Process a chat request with conversation history using UNIFIED DOCUMENT SEARCH.
        Primary strategy: RAG search across ALL documents first, then MCP analysis as fallback.
        This unified approach ensures questions find best answers regardless of document type.
        """
        start_time = time.time()

        try:
            # Validate that we have messages
            if not chat_request.messages:
                raise BadRequestException("No messages provided in chat request")

            # Get accessible folders for the user
            accessible_folders = self._get_accessible_folders(user_id, chat_request.folder_ids)

            if not accessible_folders:
                raise PermissionDeniedException("No accessible folders found for query")

            # Extract latest user query
            latest_user_message = None
            for msg in reversed(chat_request.messages):
                if msg.role == "user":
                    latest_user_message = msg.content
                    break

            if not latest_user_message:
                raise BadRequestException("No user message found")

            # PRIMARY STRATEGY: Check if this is a CSV/Excel analysis question first
            # Only CSV/Excel files use MCP analysis - all other files use RAG
            is_data_analysis_query, csv_excel_files = await self._is_csv_excel_query(
                latest_user_message, accessible_folders, user_id
            )

            if is_data_analysis_query and csv_excel_files:
                print(f"No RAG results found, will use MCP analysis with {len(csv_excel_files)} available structured data files")
                for i, f in enumerate(csv_excel_files):
                    status = "(MCP ready)" if f.get('has_mcp') else "(needs processing)"
                    print(f"  [{i+1}] {f['filename']} {status}")

                # If no MCP-ready files, try to fall back to RAG with error message
                if not csv_excel_files:
                    print("No structured data files available")
                    return ChatResponse(
                        role="assistant",
                        content="I couldn't find any structured data files (CSV/Excel) for analysis. Please upload some data files first.",
                        sources=[],
                        total_chunks=0,
                        processing_time=time.time() - start_time,
                        reformulated_query=latest_user_message
                    )

                try:
                    print("MCP Analysis: Starting MCP data analysis...")
                    print(f"MCP Analysis: Available files: {len(csv_excel_files)}")
                    for i, f in enumerate(csv_excel_files):
                        cols = f.get('columns', [])
                        print(f"MCP Analysis: File {i+1}: {f['filename']} - has_mcp: {f.get('has_mcp', False)}, columns: {len(cols)} ({', '.join(cols[:3])}{'...' if len(cols) > 3 else ''})")

                    print("MCP Analysis: Creating EnhancedMCPTools instance")
                    from app.mcp.enhanced_tools import EnhancedMCPTools
                    enhanced_mcp_tools = EnhancedMCPTools(self.db, str(user_id))
                    print("MCP Analysis: EnhancedMCPTools created successfully")

                    # Ensure MCP processor has database access for persistent metadata
                    if not enhanced_mcp_tools.data_processor.db:
                        enhanced_mcp_tools.data_processor.db = self.db
                        print("MCP Analysis: Database session assigned to MCP processor")
                    else:
                        print("MCP Analysis: MCP processor already has database access")

                    # Create context for pandas code generation (with safe column access)
                    files_context_lines = []
                    for f in csv_excel_files:
                        col_list = f.get('columns', [])
                        col_desc = ', '.join(col_list[:5])
                        if len(col_list) > 5:
                            col_desc += '...'
                        files_context_lines.append(f"- {f['filename']}: columns {col_desc} ({len(col_list)} total)")

                    files_context = "\n".join(files_context_lines)
                    print(f"MCP Analysis: Created file context with {len(files_context_lines)} files")

                    # Build chat history context (like POC approach)
                    chat_history_context = ""
                    if len(chat_request.messages) > 1:
                        recent_entries = chat_request.messages[-3:]  # Last 3 conversations
                        chat_history_context = "\nRecent Conversation:\n" + "\n".join([
                            f"Q: {msg.content}\nA: Previous analysis completed"
                            for msg in recent_entries[:-1]  # Exclude the latest question
                        ]) + "\n"

                    # Use AI to determine which file is most relevant for the question
                    target_file = await self._select_best_file_for_question(
                        question=latest_user_message,
                        available_files=csv_excel_files
                    )

                    if not target_file:
                        # Fallback to first file if AI selection fails
                        target_file = csv_excel_files[0]

                    # DEBUG: Check if target file has columns before proceeding
                    file_columns = target_file.get('columns', [])
                    print(f"MCP Analysis: Selected file: {target_file['filename']} with {len(file_columns)} columns: {file_columns[:5]}{'...' if len(file_columns) > 5 else ''}")

                    # FORCE column loading - try multiple methods to get columns
                    if not file_columns:
                        print("MCP Analysis: No columns found in file info, trying to load them...")

                        # Method 1: Try to read directly from database metadata
                        try:
                            from app.models import McpFileMetadata
                            mcp_file_id = target_file.get('file_id') or target_file.get('mcp_file_id')
                            if mcp_file_id:
                                db_file = self.db.query(McpFileMetadata).filter(McpFileMetadata.file_id == mcp_file_id).first()
                                if db_file and db_file.columns:
                                    file_columns = db_file.columns
                                    print(f"MCP Analysis: ✓ Loaded {len(file_columns)} columns from database metadata")
                        except Exception as db_err:
                            print(f"MCP Analysis: ✗ Database metadata load failed: {db_err}")

                        # Method 2: Try direct file reading via MCP processor
                        if not file_columns:
                            try:
                                file_id = target_file.get('file_id', f"{target_file['user_id']}_{target_file['folder_id']}_{target_file['filename']}")
                                columns = await enhanced_mcp_tools.data_processor.get_columns(file_id)
                                if columns:
                                    file_columns = columns
                                    target_file['columns'] = columns
                                    print(f"MCP Analysis: ✓ Loaded {len(file_columns)} columns by direct file reading")
                            except Exception as col_err:
                                print(f"MCP Analysis: ✗ Direct file reading failed: {col_err}")

                        # Method 3: Attempt to create file_id and read (for files uploaded after fixes)
                        if not file_columns:
                            try:
                                # Create the expected file_id format
                                expected_file_id = f"{target_file['user_id']}_{target_file['folder_id']}_{target_file['filename']}"
                                columns = await enhanced_mcp_tools.data_processor.get_columns(expected_file_id)
                                if columns:
                                    file_columns = columns
                                    target_file['columns'] = columns
                                    target_file['file_id'] = expected_file_id
                                    print(f"MCP Analysis: ✓ Loaded {len(file_columns)} columns using expected file_id: {expected_file_id}")
                            except Exception as exp_err:
                                print(f"MCP Analysis: ✗ Expected file_id method failed: {exp_err}")

                        if file_columns:
                            print(f"MCP Analysis: SUCCESS - File now has {len(file_columns)} columns available for analysis")
                        else:
                            print("MCP Analysis: FAILURE - Could not load columns for data analysis")

                    # Now proceed with analysis if we have any columns OR try columnless analysis
                    can_proceed = bool(file_columns)
                    print(f"MCP Analysis: Can proceed with analysis: {can_proceed} (columns: {len(file_columns) if file_columns else 0})")

                    if not can_proceed:
                        fallback_response = await self._generate_fallback_response(
                            question=latest_user_message,
                            files_info=csv_excel_files
                        )

                        return ChatResponse(
                            role="assistant",
                            content=f"I'm sorry, but I can't analyze '{latest_user_message}' with the available data files. The selected file ({target_file['filename']}) doesn't have accessible column information for analysis.\n\nFallback: {fallback_response}",
                            sources=[],
                            total_chunks=0,
                            processing_time=time.time() - start_time,
                            reformulated_query=latest_user_message
                        )

                    # ENHANCED QUESTION PROCESSING: Deep understanding and reframing for better analysis
                    print("MCP Analysis: Deeply understanding the question and reframing for data analysis...")

                    # Add question preprocessing to better understand and map to columns
                    enhanced_question = await self._preprocess_mcp_question(latest_user_message, target_file)
                    print(f"MCP Analysis: ✓ Question reframed: '{enhanced_question}'")

                    # Generate pandas code using the enhanced question and selected file
                    target_file_list = [target_file]  # Pass as list for consistency

                    print(f"MCP Analysis: 🧠 Generating pandas code for: '{enhanced_question}'")
                    pandas_generated = await self._generate_pandas_code(
                        question=enhanced_question,  # Use enhanced question for better results
                        files_context="",  # Will be set inside
                        chat_history_context=chat_history_context,
                        available_files=target_file_list  # Use only the selected file
                    )

                    print(f"MCP Analysis: ✓ Pandas code generated: {pandas_generated.get('description', 'N/A')}")

                    if pandas_generated and "error" not in pandas_generated:
                        print(f"MCP Analysis: 🔄 Executing: {pandas_generated['pandas_code']}")
                        # Execute the generated pandas operations on the selected file
                        result = await enhanced_mcp_tools.query_data(
                            file_id=target_file["file_id"],
                            operation="execute",
                            code=pandas_generated["pandas_code"],
                            session_id=f"mcp_session_{int(time.time())}",
                            question=latest_user_message
                        )

                        print(f"MCP Analysis: ✅ Raw result obtained: {type(result)}")

                        # Format natural response from raw result
                        natural_response = await self._format_mcp_response(
                            question=latest_user_message,
                            raw_result=result.get("result"),
                            filename=target_file["filename"]
                        )

                        print(f"MCP Analysis: 🎨 Response formatted: {len(natural_response)} characters")

                        return ChatResponse(
                            role="assistant",
                            content=natural_response,
                            sources=[RAGChunk(
                                document_id=target_file["file_id"],
                                document_name=target_file["filename"],
                                folder_id=target_file["folder_id"],
                                folder_name=f"Folder {target_file['folder_id']}",
                                chunk_text=f"MCP Data Analysis: {pandas_generated.get('description', 'Analysis performed')}",
                                relevance_score=1.0,
                                metadata={"analysis_type": "direct_mcp", "pandas_code": pandas_generated["pandas_code"]}
                            )],
                            total_chunks=1,
                            processing_time=time.time() - start_time,
                            reformulated_query=latest_user_message
                        )
                    else:
                        print(f"MCP Analysis: ❌ Pandas code generation failed: {pandas_generated}")
                        # Fallback response
                        fallback_response = await self._generate_fallback_response(
                            question=latest_user_message,
                            files_info=csv_excel_files
                        )

                        return ChatResponse(
                            role="assistant",
                            content=fallback_response,
                            sources=[],
                            total_chunks=0,
                            processing_time=time.time() - start_time,
                            reformulated_query=latest_user_message
                        )

                except Exception as e:
                    print(f"MCP analysis failed: {str(e)}")
                    return ChatResponse(
                        role="assistant",
                        content="I wasn't able to analyze the data using direct methods. Let me try regular document search instead.",
                        sources=[],
                        total_chunks=0,
                        processing_time=time.time() - start_time,
                        reformulated_query=latest_user_message
                    )

            else:
                # **Regular Document Queries - Use RAG**
                print(f"RAG Pipeline: Processing DOCUMENT question: '{latest_user_message}'")

                try:
                    # Check if we have any documents with embeddings
                    from app.models import Embedding
                    total_embeddings = self.db.query(Embedding).filter(
                        Embedding.document_id.in_([
                            doc.id for doc in self.db.query(Document)
                            .filter(Document.folder_id.in_(accessible_folders))
                            .all()
                        ])
                    ).count()
                    print(f"RAG Pipeline: Found {total_embeddings} total embeddings across {len(accessible_folders)} folders")

                    if total_embeddings == 0:
                        print("RAG Pipeline: No embeddings found - no documents processed yet")
                        return ChatResponse(
                            role="assistant",
                            content="No documents have been processed yet for semantic search. Please upload some PDF, DOCX, or text documents first, or ask questions about your data files.",
                            sources=[],
                            total_chunks=0,
                            processing_time=time.time() - start_time,
                            reformulated_query=latest_user_message
                        )

                except Exception as e:
                    print(f"RAG Pipeline: Error checking embeddings: {str(e)}")

                # Take last 5 messages for context window
                context_window_size = 5
                recent_messages = chat_request.messages[-context_window_size:] if len(chat_request.messages) > context_window_size else chat_request.messages

                print(f"RAG Pipeline: Reformulating query...")
                # Reformulate the latest query based on conversation history
                reformulated_query = await self._reformulate_query(recent_messages)
                print(f"RAG Pipeline: Original: '{latest_user_message}' → Reformulated: '{reformulated_query}'")

                try:
                    print(f"RAG Pipeline: Generating embedding for query...")
                    # Generate query embedding using reformulated query
                    query_embedding = self.embedding_service.generate_embeddings([reformulated_query])[0]
                    print(f"RAG Pipeline: Generated embedding ({len(query_embedding)} dimensions)")

                    print(f"RAG Pipeline: Searching for similar chunks...")
                    # Search for similar chunks from all embedded documents
                    # Lower similarity threshold for better recall (0.3 instead of 0.7 default)
                    min_relevance = min(chat_request.min_relevance_score or 0.3, 0.3)  # At least 0.3 for better recall
                    similar_chunks = self.embedding_service.search_similar_chunks(
                        query_embedding=query_embedding,
                        folder_ids=accessible_folders,
                        limit=max(chat_request.limit, 10),  # Ensure at least 10 results
                        min_similarity=min_relevance
                    )
                    print(f"RAG Pipeline: Found {len(similar_chunks)} similar chunks")

                except Exception as e:
                    print(f"RAG Pipeline: CRITICAL ERROR in search: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise BadRequestException(f"Document search failed: {str(e)}")

                if not similar_chunks:
                    return ChatResponse(
                        role="assistant",
                        content="No relevant documents found for your query. Try uploading documents first or rephrase your question.",
                        sources=[],
                        total_chunks=0,
                        processing_time=time.time() - start_time,
                        reformulated_query=reformulated_query
                    )

                try:
                    print(f"RAG Pipeline: Generating final answer using {len(similar_chunks)} document chunks...")
                    # Generate answer using RAG for regular queries
                    answer = await self._generate_rag_answer(recent_messages, similar_chunks)
                    print(f"RAG Pipeline: Generated answer ({len(answer)} characters)")
                except Exception as e:
                    print(f"RAG Pipeline: ERROR generating answer: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise BadRequestException(f"Failed to generate RAG answer: {str(e)}")

                # Format sources
                sources = []
                for chunk in similar_chunks:
                    try:
                        source = RAGChunk(
                            document_id=str(chunk["document_id"]),  # Convert UUID to string
                            document_name=chunk["document_name"],
                            folder_id=str(chunk["folder_id"]),     # Convert UUID to string
                            folder_name=chunk["folder_name"],
                            chunk_text=chunk["chunk_text"],
                            relevance_score=chunk["similarity_score"],
                            metadata=chunk["metadata"]
                        )
                        sources.append(source)
                    except Exception as e:
                        print(f"RAG Pipeline: WARNING - Error creating source chunk: {e}")
                        continue

                processing_time = time.time() - start_time

                print(f"RAG Pipeline: Final response - {len(sources)} sources, {processing_time:.2f}s processing time")

                try:
                    response_obj = ChatResponse(
                        role="assistant",
                        content=answer,
                        sources=sources,
                        total_chunks=len(similar_chunks),
                        processing_time=processing_time,
                        reformulated_query=reformulated_query
                    )
                    print(f"RAG Pipeline: Successfully created ChatResponse object")
                    return response_obj
                except Exception as e:
                    print(f"RAG Pipeline: CRITICAL ERROR - Failed to create ChatResponse: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise BadRequestException(f"Failed to format response: {str(e)}")

        except Exception as e:
            if isinstance(e, (BadRequestException, PermissionDeniedException)):
                raise
            raise BadRequestException(f"Failed to process chat request: {str(e)}")

    async def _classify_question_relevance(self, question: str, available_csv_excel_files: List[Dict], chat_history: List[ChatMessage] = None) -> dict:
        """
        Classify if a question is relevant to the uploaded data files using STRICT column matching.
        Enhanced to detect aggregation patterns like "region wise sales".
        Returns {'is_relevant': bool, 'reason': str, 'query_type': str}
        """
        try:
            if not available_csv_excel_files:
                return {
                    "is_relevant": False,
                    "reason": "No CSV/Excel files available for analysis",
                    "query_type": "none"
                }

            # Analyze question structure
            question_lower = question.lower()
            question_analysis = self._analyze_question_structure(question_lower)

            print(f"DEBUG: Question analysis: {question_analysis}")

            # Extract potential column entities from question
            question_entities = []

            # Common question keywords that indicate data analysis
            analysis_keywords = ['top', 'most', 'highest', 'lowest', 'average', 'mean', 'sum', 'count', 'min', 'max', 'total', 'best', 'wise']
            entity_keywords = ['region', 'city', 'state', 'country', 'category', 'product', 'customer', 'sales', 'profit', 'revenue', 'quantity', 'price']

            # Extract potential column entities from question (excluding common question words)
            question_words = question_lower.replace('?', '').replace('!', '').split()
            for word in question_words:
                if len(word) > 2 and word not in ['the', 'and', 'for', 'with', 'from', 'what', 'which', 'who', 'how', 'when', 'where', 'why', 'are', 'was', 'were', 'has', 'have', 'had', 'but', 'can', 'did', 'does', 'had', 'has', 'how', 'may', 'might', 'must', 'need', 'shall', 'should', 'will', 'would', 'give', 'me', 'data', 'information']:
                    question_entities.append(word)

            # Question entities + analysis keywords give us context clues
            relevance_clues = question_entities + analysis_keywords + question_analysis.get('key_terms', [])

            print(f"DEBUG: Question entities extracted: {question_entities}")
            print(f"DEBUG: Relevance clues: {relevance_clues}")
            print(f"DEBUG: Query type: {question_analysis['query_type']}")

            # Check each file for column matching
            file_matches = {}
            for file_info in available_csv_excel_files:
                filename = file_info['filename']
                columns = [col.lower() for col in file_info['columns']]
                filename_lower = filename.lower()

                # Score matches based on column names and filename relevance
                score = 0
                matched_terms = []

                for clue in relevance_clues:
                    for col in columns:
                        if clue in col or any(clue.endswith(word) or clue.startswith(word) for word in col.split('_')):
                            score += 2  # Column match is strong evidence
                            matched_terms.append(f"{col} (column)")
                            break

                    # Filename matching (slightly less weight)
                    if clue in filename_lower:
                        score += 1
                        matched_terms.append(f"{filename} (filename)")

                file_matches[filename] = {
                    'score': score,
                    'matched_terms': matched_terms,
                    'columns': file_info['columns']
                }

            # Find best matching file
            best_match = max(file_matches.items(), key=lambda x: x[1]['score'])
            best_score = best_match[1]['score']
            best_filename = best_match[0]

            print(f"DEBUG: File relevance scores: {[(f, s['score']) for f, s in file_matches.items()]}")
            print(f"DEBUG: Best match: {best_filename} (score: {best_score}, terms: {best_match[1]['matched_terms']})")

            # STRICT THRESHOLD: Only consider relevant if we have solid matches
            if best_score >= 2:  # At least one column match or multiple filename matches
                return {
                    "is_relevant": True,
                    "reason": f"Question can be answered using {best_filename} - matched terms: {', '.join(best_match[1]['matched_terms'])}",
                    "query_type": question_analysis['query_type'],
                    "aggregation_info": question_analysis.get('aggregation_info', {})
                }
            else:
                # NO MATCHES FOUND - this question doesn't appear to be answerable by any uploaded files
                return {
                    "is_relevant": False,
                    "reason": f"No relevant data found in uploaded files. Question appears to be about {question_entities[:3]} but available files have different data",
                    "query_type": "none"
                }

        except Exception as e:
            print(f"Error in question relevance classification: {str(e)}")
            # Default to NOT relevant on error (safer than assuming relevance)
            return {
                "is_relevant": False,
                "reason": f"Classification failed: {str(e)}",
                "query_type": "none"
            }

    async def _is_csv_excel_query(self, query: str, accessible_folders: List[UUID], user_id: UUID) -> tuple[bool, List[Dict]]:
        """
        SIMPLE OpenAI-based detection: Analyze user question to determine if it's about structured data (CSV/Excel).
        For unstructured questions (documents) → use RAG
        For structured questions (numbers, aggregates, CSV/Excel data) → use MCP tools

        **IMPROVED**: Now includes ALL CSV/Excel documents, not just those with MCP metadata.
        Files without MCP metadata will be processed on-demand when needed.
        """
        # Find ALL CSV/Excel files in accessible folders (both with and without MCP metadata)
        all_csv_excel_docs = []
        mcp_files = []

        try:
            from app.models import Document
            csv_excel_docs = self.db.query(Document).filter(
                Document.folder_id.in_(accessible_folders),
                Document.file_type.in_(['csv', 'xlsx', 'xls'])
            ).all()

            print(f"Found {len(csv_excel_docs)} CSV/Excel docs in database")

            for doc in csv_excel_docs:
                # Create info for ALL CSV/Excel files
                file_info = {
                    'document_id': doc.id,
                    'filename': doc.filename,
                    'folder_id': str(doc.folder_id),
                    'user_id': str(doc.uploaded_by),
                    'has_mcp': bool(doc.doc_metadata and doc.doc_metadata.get('mcp_file_id')),
                    'mcp_file_id': doc.doc_metadata.get('mcp_file_id') if doc.doc_metadata else None,
                    'content_analysis': doc.doc_metadata.get('content_analysis', {}) if doc.doc_metadata else {}
                }

                # If it has MCP metadata, try to get full details
                if file_info['has_mcp']:
                    mcp_file_id = doc.doc_metadata['mcp_file_id']
                    try:
                        from app.mcp.data_processor import MCPDataProcessor
                        from app.config import settings
                        mcp_processor = MCPDataProcessor(settings, self.db)  # Pass database session

                        user_folder_files = await mcp_processor.list_user_files(str(user_id))
                        matching_file = next(
                            (f for f in user_folder_files if f['file_id'] == mcp_file_id),
                            None
                        )

                        if matching_file:
                            # Add MCP-specific data to the file info
                            file_info.update({
                                'file_id': matching_file['file_id'],
                                'columns': matching_file['columns'],
                                'row_count': matching_file['row_count']
                            })
                            mcp_files.append(file_info)
                            all_csv_excel_docs.append(file_info)  # Add enriched version to all files
                        else:
                            # MCP file exists in doc metadata but not found in MCP processor
                            print(f"Warning: MCP file {mcp_file_id} not found in MCP processor")
                            all_csv_excel_docs.append(file_info)
                    except Exception as e:
                        print(f"Error checking MCP file {mcp_file_id}: {e}")
                        all_csv_excel_docs.append(file_info)
                else:
                    # No MCP metadata, try to get basic column info from the file
                    try:
                        from app.mcp.data_processor import MCPDataProcessor
                        from app.config import settings
                        mcp_processor = MCPDataProcessor(settings, self.db)

                        # Init the processor with DB to sync metadata
                        await mcp_processor.sync_with_database(self.db)

                        # Create a temp file_id based on the file path for processing
                        temp_file_id = f"{user_id}_{doc.folder_id}_{doc.filename}"

                        # Get column info if available
                        try:
                            columns = await mcp_processor.get_columns(temp_file_id)
                            file_info['columns'] = columns
                            file_info['row_count'] = await mcp_processor.read_file(temp_file_id)
                            file_info['row_count'] = len(file_info['row_count']) if hasattr(file_info['row_count'], '__len__') else 0
                        except Exception as col_err:
                            print(f"Could not get columns for {doc.filename}: {col_err}")
                            file_info['columns'] = []  # Default empty list

                        all_csv_excel_docs.append(file_info)
                    except Exception as proc_err:
                        print(f"Error processing non-MCP file {doc.filename}: {proc_err}")
                        file_info['columns'] = []  # Default empty list
                        all_csv_excel_docs.append(file_info)

        except Exception as e:
            print(f"Error checking CSV/Excel files: {e}")

        if not all_csv_excel_docs:
            print("No CSV/Excel files found in accessible folders")
            return False, []

        # Use OpenAI to analyze question and determine if it needs structured data analysis
        try:
            # Convert files list to descriptive format for OpenAI
            files_description = []
            for file_info in all_csv_excel_docs:
                mcp_status = " (has MCP analysis)" if file_info.get('has_mcp') else " (needs MCP setup)"
                files_description.append(f"- {file_info['filename']}: {len(file_info.get('columns', []))} columns{mcp_status}")

            context = f"""
QUESTION ANALYSIS: "{query}"

AVAILABLE FILES:
{chr(10).join(files_description)}

INSTRUCTIONS: Carefully analyze this question and determine if it requires structured data analysis or document analysis.

STRUCTURED DATA ANALYSIS (CSV/Excel) when question asks about:
- NUMBERS: top 5, most, highest, lowest, averages, sums, totals, counts, statistics
- FILTERING: by region/state/city, specific categories, dates, ranges
- COMPARISONS: compare regions, find differences, rankings, trends
- GROUPING: group by category, sales by region, aggregate operations
- QUANTITATIVE: "how many", "what's the average", "find max/min", "calculate sum"
- EXAMPLES: "top performers", "sales by region", "average price", "most profitable"

DOCUMENT ANALYSIS (PDF/Text) when question asks about:
- EXPLANATIONS: explain concepts, describe processes, what is/how does, theory
- CONCEPTS: understand ideas, discuss topics, analyze procedures
- CONTENT: what's mentioned, describe something, summarize information
- MEANING: interpret text, understand context, analyze meaning
- EXAMPLES: "explain cybersecurity", "what does it mean", "describe the process"

IMPORTANT OVERRIDING RULES:
- FIRST PRIORITY: Questions asking "what is X?", "explain Y?", "what does Z mean?", "describe X?" → ALWAYS DOCUMENT
- Concepts, definitions, explanations, understanding text → ALWAYS DOCUMENT
- Questions about data operations, calculations, statistics → STRUCTURED
- Questions mentioning specific column names, filtering data → STRUCTURED

FILE CONTEXT IS SECONDARY: Available CSV/Excel files do NOT change classification - the question semantic meaning comes first.

RESPONSE: Return only "STRUCTURED" or "DOCUMENT"
"""

            analysis_response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a senior data scientist and AI researcher. You excel at understanding natural language queries and determining whether they require structured data analysis (working with numbers, aggregates, statistics, filtering) or document analysis (understanding text, concepts, procedures). Your classifications are highly accurate because you consider both semantic meaning and data operation requirements. Respond only with 'STRUCTURED' or 'DOCUMENT'."},
                    {"role": "user", "content": context}
                ],
                temperature=0.05,  # More deterministic
                max_tokens=10
            )

            classification = analysis_response.choices[0].message.content.strip().upper()

            is_structured = classification == "STRUCTURED"
            print(f"OpenAI question analysis: '{query}' → {classification}")

            if is_structured:
                print(f"🧠 QUESTION TYPE: STRUCTURED DATA → routing to MCP analysis with {len(all_csv_excel_docs)} available files")
                # Return ALL files that attempted column processing (may have empty columns if processing failed)
                return True, all_csv_excel_docs
            else:
                print(f"📚 QUESTION TYPE: DOCUMENT ANALYSIS → routing to RAG system (found {len(all_csv_excel_docs)} CSV files but ignoring for document questions)")
                return False, []

        except Exception as e:
            print(f"OpenAI classification failed: {e}, falling back to keyword detection")
            # Simple fallback: look for data analysis keywords in question
            data_keywords = ['average', 'sum', 'count', 'total', 'top', 'highest', 'lowest', 'minimum', 'maximum', 'mean', 'median', 'filter', 'sort', 'aggregate', 'group by', 'wise']
            question_lower = query.lower()

            has_data_keywords = any(keyword in question_lower for keyword in data_keywords)

            if has_data_keywords and all_csv_excel_docs:
                print(f"Keyword fallback: Found data keywords in '{query}' → using structured analysis")
                # Prefer MCP-ready files, fall back to all files
                return True, mcp_files or all_csv_excel_docs
            else:
                print(f"Keyword fallback: No data keywords in '{query}' → defaulting to document analysis")
                return False, []

    async def _select_best_file_for_question(
        self,
        question: str,
        available_files: List[Dict]
    ) -> Optional[Dict]:
        """
        DIRECT COLUMN-BASED FILE SELECTION:
        Match question keywords against actual column names in files.
        More reliable than AI for file selection.
        """
        if len(available_files) == 1:
            return available_files[0]

        if not available_files:
            return None

        question_lower = question.lower()

        # Extract key terms from question (more aggressive extraction)
        key_terms = set()
        question_words = question_lower.replace('?', '').replace('!', '').split()

        # Keep all meaningful words (longer than 2 chars, not stop words)
        stop_words = {'the', 'and', 'for', 'with', 'from', 'what', 'which', 'who', 'how', 'when', 'where', 'why',
                     'are', 'was', 'were', 'has', 'have', 'had', 'but', 'can', 'did', 'does', 'had', 'has', 'how',
                     'may', 'might', 'must', 'need', 'shall', 'should', 'will', 'would', 'give', 'me', 'data', 'information'}

        for word in question_words:
            # Include words that are clearly column-like (contain numbers, underscores, etc.)
            if len(word) > 2 and word not in stop_words:
                key_terms.add(word)

        # ALSO include terms that look like column names
        for word in question_words:
            if '_' in word or any(c.isdigit() for c in word):
                key_terms.add(word)

        print(f"Searching files for question terms: {key_terms}")

        # Score each file based on column matches
        file_scores = {}
        for file_info in available_files:
            filename = file_info['filename']
            columns = [col.lower() for col in file_info['columns']]

            score = 0
            matched_columns = []

            for term in key_terms:
                # Direct column name match (strongest signal)
                for col in columns:
                    if term in col or col in term:  # Bidirectional matching
                        score += 10
                        matched_columns.append(f"{term}→{col}")
                        break

                # Filename match (moderate signal)
                if term in filename.lower():
                    score += 2
                    matched_columns.append(f"{term}→filename")

            file_scores[filename] = {
                'score': score,
                'matched_columns': matched_columns,
                'file_info': file_info
            }

            print(f"  {filename}: score={score}, matches={matched_columns}")

        # Find highest scoring file
        if file_scores:
            best_filename = max(file_scores.items(), key=lambda x: x[1]['score'])[0]
            best_score = file_scores[best_filename]['score']

            print(f"BEST MATCH: {best_filename} (score: {best_score})")

            # If we have any reasonable match (score > 0), use it
            if best_score > 0:
                return file_scores[best_filename]['file_info']

        # Fallback: return first available file
        print("No good matches found, using first available file")
        return available_files[0]

    async def _generate_rag_answer(
        self,
        messages: List[ChatMessage],
        context_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generate RAG answer using conversation history and retrieved context.
        """
        # Prepare context from chunks
        context_texts = []
        for chunk in context_chunks:
            context_texts.append(
                f"Document: {chunk['document_name']}\n"
                f"Content: {chunk['chunk_text']}\n"
                f"Relevance: {chunk['similarity_score']:.2f}\n"
            )

        document_context = "\n---\n".join(context_texts)

        # Create system prompt for RAG responses
        system_prompt = """You are a helpful AI assistant that answers questions based on provided documents and conversation history.

Use the provided document context to answer questions accurately. Maintain conversation continuity by considering the chat history.
If the document context doesn't contain enough information to answer the question, say so clearly.
Cite the relevant documents when possible.
Be conversational and natural in your responses."""

        # Build messages for OpenAI
        openai_messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (excluding system messages from user)
        for msg in messages[:-1]:  # Exclude the last message initially
            if msg.role != "system":
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        # Add the latest message with document context
        latest_message = messages[-1]
        if latest_message.role == "user":
            enhanced_content = f"""Based on the following document context, please answer this question:

Question: {latest_message.content}

Document Context:
{document_context}

Answer:"""
            openai_messages.append({
                "role": "user",
                "content": enhanced_content
            })

        try:
            response = self.openai_client.chat.completions.create(
                model=settings.openai_chat_model,
                messages=openai_messages,
                max_tokens=500,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            raise BadRequestException(f"Failed to generate RAG chat answer: {str(e)}")

    async def _generate_pandas_code(
        self,
        question: str,
        files_context: str,
        chat_history_context: str,
        available_files: List[Dict]
    ) -> Dict[str, Any]:
        """Generate pandas code to answer any question about any structured data file"""
        try:
            # Create context with available files and their columns
            available_columns = {}
            for file_info in available_files:
                available_columns[file_info['filename']] = file_info['columns']

            # Create a comprehensive context showing exact column names
            column_examples = []
            for filename, columns in available_columns.items():
                column_examples.append(f"File '{filename}' columns: {', '.join([f'"{col}"' for col in columns[:15]])}")

            files_context_json = str(available_columns)
            columns_context = "\n".join(column_examples)

            # Enhanced question preprocessing using AI to understand the question deeply
            enhanced_question = await self._enhance_question_understanding(question, columns_context, chat_history_context)

            context = f"""{chat_history_context}

QUESTION ENHANCEMENT: "{question}" → Enhanced Analysis: "{enhanced_question}"

AVAILABLE DATA FILES:
{columns_context}

INSTRUCTIONAL GUIDANCE:
You are an expert data scientist analyzing any type of structured data (CSV, Excel, etc.).
Your job is to understand questions deeply, select the best available data file, and generate precise pandas operations.

QUESTION ANALYSIS PROCESS:
1. Identify what the user wants (aggregate, filter, sort, analyze patterns, etc.)
2. Map user terms to AVAILABLE column names from the files above
3. Choose appropriate pandas operations for the data type
4. Generate executable pandas code using EXACT column names

GENERAL OPERATIONS BY QUESTION TYPE:

AGGREGATION ("WISE", "BY", "GROUPED"):
df.groupby('column_name')['numeric_column'].sum().sort_values(ascending=False).head(10).to_dict()

FILTRING + AGGREGATION:
df[df['column'].str.contains('value', case=False, na=False)]['numeric_column'].sum()

RANKING ("TOP", "BOTTOM", "BEST", "WORST"):
df.groupby('group_col')['measure_col'].sum().sort_values(ascending=False).head(N).to_dict()

STATISTICS ("AVERAGE", "MEAN", "MIN", "MAX"):
df['numeric_column'].mean() OR df.groupby('category')['numeric_col'].mean()

COUNTS ("HOW MANY", "COUNT"):
len(df) OR df['column'].count() OR df.groupby('column').size()

EXAMPLES FOR ANY DATA TYPE:
- "sales by region" → df.groupby('region_column')['sales_column'].sum().to_dict()
- "average price by category" → df.groupby('category_column')['price_column'].mean().sort_values(ascending=False).to_dict()
- "top 10 customers by revenue" → df.groupby('customer_column')['revenue_column'].sum().sort_values(ascending=False).head(10).to_dict()
- "total sales in Q4" → df[df['date_column'].str.contains('Q4|12/|QTR4', case=False, na=False)]['sales_column'].sum()
- "customer analysis" → df.groupby('customer_column').agg({{'amount_column': 'sum', 'count_column': 'count'}}).sort_values('amount_column', ascending=False).to_dict()

ANALYSIS STEPS:
1. Find the BEST file for the question based on column relevance
2. Identify the EXACT column names to use (must match what's shown above)
3. Select appropriate operation based on question intent
4. Generate pandas code that works with any data structure

RESPONSE FORMAT: JSON only
{{
  "filename": "exact_filename",
  "pandas_code": "valid_pandas_operation_using_exact_columns",
  "description": "clear_explanation_of_analysis"
}}

Example patterns for the enhanced question analysis:
- "region wise sales" anywhere → df.groupby('region/state/country_column')['sales/revenue/amount_column'].sum().sort_values(ascending=False).head(10).to_dict()
- "product performance" → df.groupby('product/item/name_column')['sales/units/revenue_column'].sum().sort_values(ascending=False).head(10).to_dict()
- "customer insights" → df.groupby('customer/client/name_column')['purchase/revenue/amount_column'].sum().sort_values(ascending=False).head(10).to_dict()

The key is FINDING the right column names and applying appropriate operations based on data types."""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a senior data scientist who can analyze ANY structured data file. You deeply understand natural language questions, map terms to actual column names, and generate precise pandas operations. Respond ONLY with valid JSON using exact column names from the data."},
                    {"role": "user", "content": context}
                ],
                temperature=0.1,
                max_tokens=300
            )

            ai_response = response.choices[0].message.content.strip()

            # Clean up response
            if ai_response.startswith("```"):
                ai_response = ai_response.split("```")[1]
                if ai_response.startswith("json"):
                    ai_response = ai_response[4:]
                ai_response = ai_response.strip()

            result = json.loads(ai_response)

            # Validate that the filename matches an available file
            if result.get('filename') not in available_columns:
                # Try to find the closest match or default to first file
                available_filenames = list(available_columns.keys())
                if available_filenames:
                    result['filename'] = available_filenames[0]
                else:
                    return {"error": "No available files found"}

            return result

        except Exception as e:
            print(f"Error generating universal pandas code: {str(e)}")
            return {"error": f"Failed to generate pandas code: {str(e)}"}

    async def _enhance_question_understanding(
        self,
        question: str,
        available_data_context: str,
        chat_history_context: str
    ) -> str:
        try:
            analysis_context = f"""{chat_history_context}
User Question: "{question}"

Available Data Files and Columns:
{available_data_context}

Your task is to deeply analyze this question and enhance it for structured data analysis.

ANALYSIS STEPS:
1. Understand the question intent (aggregate, filter, analyze, compare, etc.)
2. Identify what data relationships or patterns are being asked about
3. Map question terms to potential column names
4. Determine the appropriate analysis type

QUESTION PATTERNS TO RECOGNIZE:
- "X wise Y" means "group by X, aggregate Y" (sales by region, profit by category)
- "X by Y" means "Y grouped by X" (products by category, sales by month)
- "top/bottom X by Y" means ranking (top customers by revenue)
- "analysis of X" means comprehensive grouping + aggregation
- "how many X" means counting group sizes or totals
- "average X by Y" means mean aggregation by groups

ENHANCED ANALYSIS EXAMPLES:
- "region wise sales" → "Aggregate total sales grouped by region (find region and sales columns)"
- "customer performance" → "Analyze customer behavior by grouping customers and aggregating purchase metrics"
- "product analysis by category" → "Group products by category and calculate performance metrics"
- "monthly trends" → "Group data by time periods and analyze changes over time"

Provide a detailed analytical question that shows deep understanding of the user's intent."""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analysis expert who understands business questions deeply. Reformulate user questions to show analytical intent and specify what data operations are needed. Focus on understanding relationships and patterns in the data."},
                    {"role": "user", "content": analysis_context}
                ],
                temperature=0.2,
                max_tokens=200
            )

            enhanced = response.choices[0].message.content.strip()
            return enhanced if len(enhanced) > 10 else question  # Fallback to original

        except Exception as e:
            print(f"Question enhancement failed: {str(e)}")
            return question  # Always fallback to original question

    async def _format_mcp_response(
        self,
        question: str,
        raw_result: Any,
        filename: str
    ) -> str:
        """Format raw MCP tool results into natural language response"""
        try:
            context = f"""Dataset: {filename}
Raw Query Result: {raw_result}
Question: {question}

Based on the raw result above, provide a clear, natural language answer to the user's question. Use the data to give specific insights and format numbers appropriately. Be conversational and informative."""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analysis assistant that converts raw data query results into clear, natural language answers that business users can understand. Be direct and informative. Format numbers appropriately and explain what the data represents."},
                    {"role": "user", "content": context}
                ],
                temperature=0.3,
                max_tokens=300
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error formatting MCP response: {str(e)}")
            return f"Analysis completed. Raw result: {raw_result}"

    async def _preprocess_mcp_question(
        self,
        question: str,
        target_file: Dict
    ) -> str:
        """
        PREPROCESS MCP QUESTIONS for better column mapping and pandas code generation.
        Makes questions more explicit about what operations to perform and which columns to use.

        Example transformations:
        - "east, west, north, south region sales" → "What are the total sales for each region in the East, West, North, and South regions in the 'Country/Region' column?"
        - "top 5 batters with most runs scored" → "Show the top 5 players with the highest values in the 'total_runs' column"
        """
        try:
            columns = target_file.get('columns', [])
            filename = target_file.get('filename', '')

            # Get file column information for context
            available_columns = [f'"{col}"' for col in columns[:15]]  # Show first 15 columns max

            # Analyze question structure to understand what operation is needed
            question_lower = question.lower()
            analysis = self._analyze_question_structure(question_lower)

            context = f"""Question: "{question}"
File: "{filename}"
Available Columns: {', '.join(available_columns)}

This is a structured data analysis question. Transform the user's natural language question into a more explicit analytical query that clearly specifies:
1. What column(s) to analyze (must use EXACT column names from the available columns)
2. What mathematical operation to perform (sum, average, count, max, min, etc.)
3. What filters or grouping to apply

Examples:
- "sales by region" → "Total sales grouped by the 'Country/Region' column"
- "west region sales" → "Sum of sales for records where the 'Country/Region' column contains 'West'"
- "east, west, north, south region sales" → "Sum of sales for each region (East, West, North, South) grouped by the 'Country/Region' column"
- "top 5 products by sales" → "Top 5 values from product column, ordered by sales column in descending order"

Respond with only the transformed question, nothing else."""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at transforming natural language data analysis questions into explicit analytical queries. Focus on clearly identifying columns, operations, and filters."},
                    {"role": "user", "content": context}
                ],
                temperature=0.2,
                max_tokens=150
            )

            enhanced_question = response.choices[0].message.content.strip()

            # Validate the transformation
            if len(enhanced_question) > len(question) * 2:  # Too verbose
                return question  # Fall back to original

            if not enhanced_question or enhanced_question.lower() == question.lower():
                return question  # No real improvement

            print(f"MCP Question Enhancement: '{question}' → '{enhanced_question}'")
            return enhanced_question

        except Exception as e:
            print(f"MCP question preprocessing failed: {str(e)}, using original question")
            return question  # Always fall back to original question on any error

    async def _generate_fallback_response(
        self,
        question: str,
        files_info: List[Dict]
    ) -> str:
        """Generate fallback response when pandas code generation fails"""
        try:
            files_context = "\n".join([
                f"- {f['filename']} ({len(f['columns'])} columns: {', '.join(f['columns'][:3])}{'...' if len(f['columns']) > 3 else ''})"
                for f in files_info
            ])

            context = f"""User asked: "{question}"

Available uploaded files:
{files_context}

I wasn't able to analyze this question directly. The question may require custom analysis that I can't automatically generate pandas code for, or there may be an issue with column name matching.

Provide a helpful response explaining the limitation and suggesting alternative approaches."""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful data analysis assistant. Politely explain when direct analysis isn't possible and suggest alternatives. Keep responses concise and encouraging."},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                max_tokens=150
            )

            return response.choices[0].message.content

        except Exception:
            return f"I'm sorry, but I couldn't analyze your question '{question}' with the current data analysis tools. Try asking simpler questions like 'what is the average of column_name?' or 'show me the first 5 rows'."
