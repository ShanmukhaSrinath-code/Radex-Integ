#!/usr/bin/env python3
"""Script to create MCP metadata for existing CSV/Excel files"""
import sys
import os
import asyncio
import json
from sqlalchemy import create_engine, text

# Add server directory to path
server_dir = os.path.join(os.getcwd(), "Radex", "server")
sys.path.insert(0, server_dir)
os.chdir(server_dir)

from app.config import settings

async def create_mcp_metadata():
    """Create MCP metadata for existing CSV/Excel files"""
    from app.mcp.data_processor import MCPDataProcessor

    # Create data processor
    mcp_processor = MCPDataProcessor(settings=settings)

    # Connect to database to find existing files
    engine = create_engine(settings.database_url)

    with engine.connect() as conn:
        try:
            print("=== Creating MCP Metadata for Existing Files ===")

            # Find all CSV/Excel documents
            result = conn.execute(text("""
                SELECT id, folder_id, filename, uploaded_by
                FROM documents
                WHERE file_type IN ('csv', 'xlsx', 'xls')
            """))
            files = result.fetchall()

            print(f"Found {len(files)} CSV/Excel files to process")

            for file_info in files:
                doc_id, folder_id, filename, user_id = file_info

                # Generate expected file_id format
                file_id = f"{user_id}_{folder_id}_{filename}"

                print(f"Processing: {filename} → {file_id}")

                try:
                    # Try to create metadata on-the-fly by calling read_file
                    # This will trigger the on-the-fly metadata creation
                    df = await mcp_processor.read_file(file_id)

                    print(f"  ✅ Created MCP metadata: {len(df.columns)} columns, {len(df)} rows")

                    # Verify it was saved to database
                    result = conn.execute(text("SELECT COUNT(*) FROM mcp_file_metadata WHERE file_id = %s"), (file_id,))
                    count = result.fetchone()[0]
                    if count > 0:
                        print(f"  ✅ Saved to database")
                    else:
                        print(f"  ❌ Not saved to database")

                except Exception as e:
                    print(f"  ❌ Failed: {str(e)}")

                print()

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(create_mcp_metadata())
