#!/usr/bin/env python3
"""
Test script for the followup bot functionality.

This script tests various scenarios to ensure the followup bot works correctly:
1. Starting a conversation with past appointments (triggers followup bot)
2. Testing medication adherence questions
3. Testing side effects inquiries
4. Testing symptom change questions
5. Testing understanding of instructions

Usage:
    python test_followup_bot.py
"""

import requests
import json
import time
from datetime import datetime, timezone, timedelta

# Configuration
BASE_URL = "http://localhost:5007"
TEST_THREAD_ID = f"test_followup_{int(time.time())}"

# Sample appointment data that should trigger followup bot
APPOINTMENT_DATA_WITH_PAST = {
    "appointments": [
        {
            "appt_id": "apt_001",
            "appt_datetime": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
            "appt_status": "completed",
            "doctor_name": "Dr. Smith",
            "symptom-summary": "Allergic reaction to peanuts",
            "prescription": "EpiPen as needed, Benadryl 25mg twice daily"
        }
    ]
}

# Sample appointment data that should NOT trigger followup bot (future appointment exists)
APPOINTMENT_DATA_WITH_FUTURE = {
    "appointments": [
        {
            "appt_id": "apt_001", 
            "appt_datetime": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
            "appt_status": "completed",
            "doctor_name": "Dr. Smith",
            "symptom-summary": "Allergic reaction to peanuts"
        },
        {
            "appt_id": "apt_002",
            "appt_datetime": (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat(),
            "appt_status": "booked", 
            "doctor_name": "Dr. Smith"
        }
    ]
}

def start_conversation(thread_id, doctor_name, appointment_data, specialty="paediatrics"):
    """Start a new conversation thread."""
    url = f"{BASE_URL}/start_conversation"
    payload = {
        "thread_id": thread_id,
        "doctor_name": doctor_name,
        "clinic_name": "Metro Allergy Clinic",
        "specialty": specialty,
        "appointment_data": appointment_data
    }
    
    response = requests.post(url, json=payload)
    print(f"Starting conversation: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def send_message(thread_id, message, appointment_data=None, prescription=None):
    """Send a message to the bot."""
    url = f"{BASE_URL}/message"
    payload = {
        "thread_id": thread_id,
        "message": message
    }
    
    if appointment_data:
        payload["appointment_data"] = appointment_data
    if prescription:
        payload["prescription"] = prescription
        
    response = requests.post(url, json=payload)
    print(f"\n--- Sending: {message} ---")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Bot reply: {data.get('reply', 'No reply')}")
        if 'bot_selection' in data:
            print(f"Bot selected: {data['bot_selection']}")
        return data.get('reply', '')
    else:
        print(f"Error: {response.json()}")
        return None

def test_followup_bot_trigger():
    """Test that followup bot is correctly triggered with past appointments."""
    print("="*60)
    print("TEST 1: Followup Bot Trigger (Past appointment, no future)")
    print("="*60)
    
    thread_id = f"{TEST_THREAD_ID}_trigger"
    
    # Start conversation with past appointments
    if start_conversation(thread_id, "Dr. Smith", APPOINTMENT_DATA_WITH_PAST):
        # Send initial message - should trigger followup bot
        reply = send_message(thread_id, "Hi, I'm here for my follow-up.")
        
        # Check if followup bot was triggered
        if reply and ("follow-up" in reply.lower() or "medication" in reply.lower()):
            print("✅ SUCCESS: Followup bot correctly triggered!")
            return thread_id
        else:
            print("❌ FAILED: Followup bot was not triggered")
            return None
    return None

def test_no_followup_trigger():
    """Test that followup bot is NOT triggered when there's a future appointment."""
    print("\n" + "="*60)
    print("TEST 2: No Followup Bot Trigger (Future appointment exists)")
    print("="*60)
    
    thread_id = f"{TEST_THREAD_ID}_no_trigger"
    
    # Start conversation with future appointments
    if start_conversation(thread_id, "Dr. Smith", APPOINTMENT_DATA_WITH_FUTURE):
        # Send initial message - should NOT trigger followup bot
        reply = send_message(thread_id, "Hi, I have some questions.")
        
        # Check if followup bot was NOT triggered (should get symptom or info bot instead)
        if reply and not ("follow-up" in reply.lower() and "medication" in reply.lower()):
            print("✅ SUCCESS: Followup bot correctly NOT triggered!")
        else:
            print("❌ FAILED: Followup bot was incorrectly triggered")

def test_followup_conversation_flow(thread_id):
    """Test the complete followup conversation flow."""
    if not thread_id:
        print("❌ Cannot test conversation flow - no valid thread ID")
        return
        
    print("\n" + "="*60)
    print("TEST 3: Followup Bot Conversation Flow")
    print("="*60)
    
    # Test medication adherence question
    print("\n--- Testing medication adherence response ---")
    reply = send_message(thread_id, "Yes, I've been taking my medications as prescribed.")
    
    # Test side effects question
    print("\n--- Testing side effects inquiry ---")
    reply = send_message(thread_id, "No side effects so far.")
    
    # Test symptom changes
    print("\n--- Testing symptom changes ---")
    reply = send_message(thread_id, "My symptoms have improved significantly.")
    
    # Test understanding of instructions
    print("\n--- Testing understanding of instructions ---")
    reply = send_message(thread_id, "Yes, I understand when to use the EpiPen.")
    
    # Test escalation scenario
    print("\n--- Testing escalation scenario ---")
    reply = send_message(thread_id, "Actually, I've been experiencing some new symptoms that worry me.")

def test_prescription_handling():
    """Test prescription data handling in followup bot."""
    print("\n" + "="*60)
    print("TEST 4: Prescription Data Handling")
    print("="*60)
    
    thread_id = f"{TEST_THREAD_ID}_prescription"
    
    # Custom prescription data
    prescription = {
        "medications": [
            {"name": "Benadryl", "dosage": "25mg", "frequency": "twice daily"},
            {"name": "EpiPen", "usage": "as needed for severe reactions"}
        ],
        "instructions": "Avoid all peanut products. Carry EpiPen at all times."
    }
    
    if start_conversation(thread_id, "Dr. Smith", APPOINTMENT_DATA_WITH_PAST):
        # Send message with prescription data
        reply = send_message(
            thread_id, 
            "I want to discuss my medications.", 
            appointment_data=APPOINTMENT_DATA_WITH_PAST,
            prescription=prescription
        )
        
        if reply:
            print("✅ Prescription data successfully processed")
        else:
            print("❌ Failed to process prescription data")

def test_error_scenarios():
    """Test error handling scenarios."""
    print("\n" + "="*60)
    print("TEST 5: Error Scenarios")
    print("="*60)
    
    # Test invalid thread ID
    print("\n--- Testing invalid thread ID ---")
    send_message("invalid_thread", "Hello")
    
    # Test missing appointment data
    print("\n--- Testing missing appointment data ---")
    thread_id = f"{TEST_THREAD_ID}_error"
    if start_conversation(thread_id, "Dr. Smith", {}):
        send_message(thread_id, "Hi there")

def run_all_tests():
    """Run all followup bot tests."""
    print("🤖 Starting Followup Bot Tests")
    print("Make sure the Flask app is running on port 5007")
    
    try:
        # Test basic connectivity
        response = requests.get(f"{BASE_URL}/")
        print("✅ Server is accessible")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure Flask app is running!")
        return
    
    # Run tests
    followup_thread = test_followup_bot_trigger()
    test_no_followup_trigger()
    test_followup_conversation_flow(followup_thread)
    test_prescription_handling()
    test_error_scenarios()
    
    print("\n" + "="*60)
    print("🎉 All tests completed!")
    print("="*60)

if __name__ == "__main__":
    run_all_tests()