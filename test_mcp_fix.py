#!/usr/bin/env python3
"""Test script to verify MCP fix works"""
import sys
import os
import asyncio
import requests
import time

# Add server directory to path
server_dir = os.path.join(os.getcwd(), "Radex", "server")
sys.path.insert(0, server_dir)
os.chdir(server_dir)

def test_mcp_chat_request():
    """Test MCP functionality via API call"""

    # First try to login to get token
    login_payload = {
        "email": "admin@radex.com",  # Default admin user
        "password": "password"
    }

    try:
        print("Logging in to get token...")
        login_response = requests.post("http://localhost:8000/api/v1/auth/login", json=login_payload)
        if login_response.status_code == 200:
            token_data = login_response.json()
            token = token_data.get("access_token")
            print("✅ Login successful")
        else:
            print(f"❌ Login failed: {login_response.status_code}, {login_response.text}")
            return False

        # Test the MCP query
        headers = {"Authorization": f"Bearer {token}"}
        chat_payload = {
            "messages": [{"role": "user", "content": "list me region wise sales"}],
            "folder_ids": [],  # Use available folders
            "max_relevance_score": 0.5,
            "limit": 10
        }

        print("Testing MCP query: 'list me region wise sales'...")
        start_time = time.time()
        chat_response = requests.post("http://localhost:8000/api/v1/chat", json=chat_payload, headers=headers)

        elapsed = time.time() - start_time
        print(f"Request took {elapsed:.2f}s")

        if chat_response.status_code == 200:
            response_data = chat_response.json()
            print("✅ Chat request successful")

            if "content" in response_data:
                content = response_data["content"]
                print(f"Response preview: {content[:300]}{'...' if len(content) > 300 else ''}")

                # Check if it's not a fallback message
                if "wasn't able to analyze" not in content.lower():
                    print("✅ MCP analysis appears to have worked!")
                    return True
                else:
                    print("❌ MCP analysis failed, got fallback message")
                    return False
            else:
                print("❌ No content in response")
                return False
        else:
            print(f"❌ Chat request failed: {chat_response.status_code}, {chat_response.text}")
            return False

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    print("Testing MCP Fix...")
    success = test_mcp_chat_request()
    if success:
        print("\n🎉 MCP Fix Test PASSED!")
    else:
        print("\n❌ MCP Fix Test FAILED!")
    sys.exit(0 if success else 1)
