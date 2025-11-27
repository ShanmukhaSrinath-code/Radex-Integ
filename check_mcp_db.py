#!/usr/bin/env python3
"""Script to check MCP file metadata in database and create missing entries"""
import sys
import os
import json
from sqlalchemy import create_engine, text

# Add server directory to path
server_dir = os.path.join(os.getcwd(), "Radex", "server")
sys.path.insert(0, server_dir)
os.chdir(server_dir)

# Import settings to get database URL
from app.config import settings

def check_mcp_database():
    """Check current state of MCP metadata in database using raw SQL"""
    engine = create_engine(settings.database_url)

    with engine.connect() as conn:
        try:
            # First check what columns exist in documents table
            print("=== Checking Database Structure ===")
            result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'documents' ORDER BY ordinal_position"))
            doc_columns = [row[0] for row in result.fetchall()]
            print(f"Documents table columns: {', '.join(doc_columns)}")

            result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'mcp_file_metadata' ORDER BY ordinal_position"))
            mcp_columns = [row[0] for row in result.fetchall()]
            print(f"MCP file metadata table columns: {', '.join(mcp_columns)}")

            # Check all documents that are CSV/Excel
            print("\n=== Checking Document Table ===")
            file_type_col = 'file_type' if 'file_type' in doc_columns else 'filename'  # fallback
            if 'file_type' in doc_columns:
                where_clause = "file_type IN ('csv', 'xlsx', 'xls')"
            else:
                # Check filename for .csv, .xlsx, .xls extensions
                where_clause = "filename LIKE '%.csv%' OR filename LIKE '%.xlsx%' OR filename LIKE '%.xls%'"

            query = f"SELECT id, filename" + (", file_type" if 'file_type' in doc_columns else "") + (" , content_metadata" if 'content_metadata' in doc_columns else "") + f" FROM documents WHERE {where_clause}"
            result = conn.execute(text(query))
            csv_excel_docs = result.fetchall()

            print(f"Found {len(csv_excel_docs)} CSV/Excel documents:")
            for doc in csv_excel_docs:
                doc_id = doc[0]
                filename = doc[1]
                file_type = doc[2] if len(doc) > 2 else "unknown"
                content_metadata = doc[3] if len(doc) > 3 else None

                print(f"  - {filename} (ID: {doc_id}, Type: {file_type})")
                if content_metadata:
                    try:
                        metadata = json.loads(content_metadata)
                        print(f"    → Content metadata: {list(metadata.keys()) if metadata else 'empty'}")
                        if 'mcp_file_id' in metadata:
                            mcp_file_id = metadata['mcp_file_id']
                            print(f"    → MCP file ID: {mcp_file_id}")
                    except Exception as e:
                        print(f"    → Invalid JSON metadata: {e}")
                else:
                    print(f"    → No content_metadata")

            print("\n=== Checking MCP File Metadata Table ===")
            if mcp_columns:
                columns_select = []
                for col in ['file_id', 'filename', 'columns', 'row_count', 'upload_time']:
                    if col in mcp_columns:
                        columns_select.append(col)

                if columns_select:
                    query = f"SELECT {', '.join(columns_select)} FROM mcp_file_metadata"
                    result = conn.execute(text(query))
                    mcp_files = result.fetchall()

                    print(f"Found {len(mcp_files)} MCP file metadata entries:")
                    for mcp_file in mcp_files:
                        row_data = dict(zip(columns_select, mcp_file))
                        print(f"  - File ID: {row_data.get('file_id', 'N/A')}")
                        print(f"    Filename: {row_data.get('filename', 'N/A')}")
                        if 'row_count' in row_data:
                            print(f"    Rows: {row_data['row_count']}")
                        if 'columns' in row_data and row_data['columns']:
                            columns_list = row_data['columns']
                            if isinstance(columns_list, str):
                                columns_list = json.loads(columns_list)
                            print(f"    Columns ({len(columns_list)}): {', '.join(columns_list[:5])}...")
                        else:
                            print(f"    Columns: EMPTY")
                        print(f"    Upload time: {row_data.get('upload_time', 'N/A')}")
                        print()
                else:
                    print("No usable columns found in MCP table")
            else:
                print("MCP file metadata table doesn't exist or has no columns")

        except Exception as e:
            print(f"Error checking database: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    check_mcp_database()
