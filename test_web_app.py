import os
import time
import threading
import requests
import uvicorn
from web_app import app


def test_web_app_suite():
    print("[TEST] Running Nova Web App & Frontend API Unit Tests...")

    # Start live server on ephemeral port 8099
    port = 8099
    server_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": "127.0.0.1", "port": port, "log_level": "warning"},
        daemon=True
    )
    server_thread.start()
    time.sleep(1.8)

    base_url = f"http://127.0.0.1:{port}"

    try:
        # 1. Test Static Index HTML serving
        r_index = requests.get(f"{base_url}/")
        assert r_index.status_code == 200, f"Expected 200, got {r_index.status_code}"
        assert "Nova AI Voice Assistant" in r_index.text
        assert "Hey Nova" in r_index.text
        print("[OK] Served Frontend Single Page Application (HTML & CSS)")

        # 2. Test Status Endpoint & Wake Word Guarantee
        r_status = requests.get(f"{base_url}/api/status")
        assert r_status.status_code == 200
        data_status = r_status.json()
        assert data_status["wake_word"] == "Hey Nova", f"Expected 'Hey Nova', got {data_status['wake_word']}"
        assert data_status["security"]["local_wake_word"] is True
        print(f"[OK] Status API verified (Wake Word: {data_status['wake_word']}, Security: {data_status['security']})")

        # 3. Test Greeting on 'Hey Nova'
        r_greet = requests.post(f"{base_url}/api/chat", json={"message": "Hey Nova"})
        assert r_greet.status_code == 200
        greet_data = r_greet.json()
        assert "listening" in greet_data["reply"].lower()
        print(f"[OK] Standalone 'Hey Nova' greeting verified: '{greet_data['reply']}'")

        # 4. Test Memory Command
        r_mem = requests.post(f"{base_url}/api/chat", json={"message": "remember that I prefer using dark theme"})
        assert r_mem.status_code == 200
        mem_data = r_mem.json()
        assert "dark theme" in mem_data["reply"]
        print(f"[OK] Direct memory storage via Web API verified: '{mem_data['reply']}'")

        # 5. Test Memory Vault Retrieval
        r_vault = requests.get(f"{base_url}/api/memory")
        assert r_vault.status_code == 200
        vault_data = r_vault.json()
        assert "profile" in vault_data
        assert "long_term_memories" in vault_data
        print(f"[OK] Memory vault inspection verified ({len(vault_data['long_term_memories'])} LTM records)")

        # 6. Test Memory Export
        r_export = requests.get(f"{base_url}/api/memory/export")
        assert r_export.status_code == 200
        assert "text/markdown" in r_export.headers.get("content-type", "")
        print("[OK] Memory export download verified")

        print("\n[PASSED] ALL NOVA WEB APP & FRONTEND UNIT TESTS PASSED SUCCESSFULLY!\n")

    finally:
        # Cleanup
        pass


if __name__ == "__main__":
    test_web_app_suite()
