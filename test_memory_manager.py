import shutil
import tempfile
from pathlib import Path
from memory_manager import MemoryManager
from security_manager import SecurityManager


def test_memory_persistence():
    print("[TEST] Running Comprehensive MemoryManager Unit Tests...")
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Step 1: Initialize first instance with standard SecurityManager
        sec = SecurityManager(enable_encryption=False)
        mem1 = MemoryManager(memory_dir=temp_dir, security_manager=sec)
        assert mem1.get_last_interaction_id() is None, "Initial interaction ID should be None"
        
        # Step 2: Auto-learning from user text
        mem1.log_turn("Hello! My name is Aman.", "Hello Aman! Nice to meet you.", interaction_id="inter_001")
        mem1.log_turn("I like Python programming and building AI assistants.", "Python is wonderful for AI!", interaction_id="inter_002")
        
        profile1 = mem1.get_user_profile()
        assert profile1["name"] == "Aman", f"Expected Aman, got {profile1.get('name')}"
        assert any("Python" in p for p in profile1.get("preferences", [])), "Preference should be learned"
        assert mem1.get_last_interaction_id() == "inter_002", "Interaction ID should be inter_002"
        
        # Step 3: Simulate application restart (new MemoryManager pointing to same directory)
        print("[RESTART] Simulating application shutdown and restart...")
        mem2 = MemoryManager(memory_dir=temp_dir, security_manager=sec)
        
        # Verify persistence across restart!
        assert mem2.get_last_interaction_id() == "inter_002", "Interaction ID must persist after restart"
        profile2 = mem2.get_user_profile()
        assert profile2["name"] == "Aman", "User name must persist after restart"
        print(f"[SUCCESS] Verified restart persistence: Name = {profile2['name']}, Last ID = {mem2.get_last_interaction_id()}")
        
        # Step 4: Test direct memory commands
        cmd_resp = mem2.handle_memory_command("What do you remember about me?")
        assert "Aman" in cmd_resp, f"Response should mention Aman: {cmd_resp}"
        print(f"[OK] Memory query command response: {cmd_resp}")
        
        # Test explicit remember command
        cmd_resp2 = mem2.handle_memory_command("remember that I am building a voice bot")
        assert "building a voice bot" in cmd_resp2
        print(f"[OK] Explicit remember response: {cmd_resp2}")
        
        # Step 5: Test Long-Term Memory (LTM) bank & Hybrid Semantic Search
        mem2.add_long_term_memory("User is developing Nova using Google Gemini API", category="work_project", tags=["nova", "gemini", "code"])
        mem2.add_long_term_memory("User drinks black coffee every morning", category="personal", tags=["coffee", "habits"])
        mem2.add_project("Voice Assistant with Wake Word and LTM")
        
        all_ltm = mem2.get_all_long_term_memories()
        assert len(all_ltm) >= 3, "LTM should store memories"
        print(f"[OK] Stored {len(all_ltm)} entries in Long-Term Memory Bank")
        
        # Test Hybrid TF-IDF / Semantic Relevance Search
        gemini_results = mem2.search_relevant_memories("Tell me about Gemini and coding", limit=2)
        assert any("Gemini" in m["content"] for m in gemini_results), "Search should find Gemini memory"
        print(f"[OK] Hybrid relevance search matched: '{gemini_results[0]['content']}'")

        # Test project query
        proj_resp = mem2.handle_memory_command("What projects am I working on?")
        assert "Voice Assistant" in proj_resp
        print(f"[OK] Project query response: {proj_resp}")
        
        # Step 6: Test Episodic Cross-Session History
        mem2.add_episodic_summary("Previous session: Designed the VoiceDetector and upgraded LTM architecture.", topics=["voice", "memory"])
        episodes = mem2.get_episodic_history(limit=1)
        assert len(episodes) == 1
        assert "VoiceDetector" in episodes[0]["summary"]
        print(f"[OK] Episodic history saved: {episodes[0]['summary']}")
        
        # Step 7: Test enriched system prompt with query grounding
        base_prompt = "You are Nova, an AI voice assistant."
        enriched = mem2.build_system_instruction(base_prompt, current_query="How do I work with Gemini?")
        assert "Aman" in enriched, "System prompt should contain user's name"
        assert "VoiceDetector" in enriched, "System prompt should contain episodic context"
        assert "Gemini" in enriched, "System prompt should contain relevant LTM"
        print("[OK] Enriched System Prompt generated correctly with dynamic memory grounding")
        
        # Step 8: Test Forget command
        cmd_forget = mem2.handle_memory_command("forget that I drink black coffee")
        assert "forgotten" in cmd_forget.lower(), f"Expected forgotten response, got: {cmd_forget}"
        remaining = mem2.get_all_long_term_memories()
        assert not any("black coffee" in m["content"] for m in remaining), "Coffee memory should be removed"
        print(f"[OK] Forget command verified: {cmd_forget}")

        # Step 9: Test Memory Export (Transparency / GDPR)
        export_file = mem2.export_memory()
        assert export_file.exists(), "Export file must exist"
        with open(export_file, "r", encoding="utf-8") as f:
            export_content = f.read()
        assert "Aman" in export_content
        assert "Nova AI Voice Assistant - Stored User Memory Export" in export_content
        print(f"[OK] Memory export verified at {export_file}")
        
        # Step 10: Test Encrypted Memory Mode
        print("\n[ENCRYPTED MODE] Testing MemoryManager with AES-256 transparent encryption...")
        enc_dir = tempfile.mkdtemp()
        try:
            sec_enc = SecurityManager(enable_encryption=True)
            mem_enc = MemoryManager(memory_dir=enc_dir, security_manager=sec_enc)
            mem_enc.set_user_name("CipherAman")
            mem_enc.add_fact("Working with encrypted local memory")
            
            # Verify file on disk is encrypted
            with open(mem_enc.profile_file, "rb") as f:
                raw_disk = f.read()
            assert not b"CipherAman" in raw_disk, "Profile file on disk must be encrypted"
            assert raw_disk.startswith(b"gAAAAA"), "Profile file must be Fernet ciphertext"

            # Verify memory manager reads it decrypted
            assert mem_enc.get_user_profile()["name"] == "Cipheraman"
            print("[OK] Transparent encrypted memory operations verified")
        finally:
            shutil.rmtree(enc_dir, ignore_errors=True)

        # Step 11: Test clear all memory
        cmd_resp3 = mem2.handle_memory_command("reset memory")
        assert mem2.get_user_profile()["name"] is None, "Name should be reset"
        assert mem2.get_last_interaction_id() is None, "Session should be reset"
        assert len(mem2.get_all_long_term_memories()) == 0, "LTM should be cleared"
        assert len(mem2.get_episodic_history()) == 0, "Episodic history should be cleared"
        print(f"[OK] Reset response: {cmd_resp3}")

        print("\n[PASSED] ALL ENHANCED MEMORY TESTS PASSED SUCCESSFULLY!\n")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_memory_persistence()
