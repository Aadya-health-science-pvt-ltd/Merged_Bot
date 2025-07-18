#!/usr/bin/env python3
"""
Interactive test script for the followup bot.

This script provides a simple chat interface to manually test the followup bot.
It automatically sets up the correct appointment data to trigger the followup bot.

Usage:
    python interactive_followup_test.py
"""

import requests
import json
import time
from datetime import datetime, timezone, timedelta

# Configuration
BASE_URL = "http://localhost:5007"
THREAD_ID = f"interactive_followup_{int(time.time())}"

# Appointment data that triggers followup bot (past appointment, no future)
FOLLOWUP_APPOINTMENT_DATA = {
    "appointments": [
        {
            "appt_id": "apt_followup_001",
            "appt_datetime": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat(),
            "appt_status": "completed",
            "doctor_name": "Dr. Sarah Johnson",
            "symptom-summary": "Seasonal allergies and asthma symptoms",
            "prescription": "Albuterol inhaler (2 puffs as needed), Claritin 10mg daily, Flonase nasal spray (2 sprays each nostril daily)"
        }
    ]
}

def start_followup_conversation():
    """Initialize a conversation that will trigger the followup bot."""
    url = f"{BASE_URL}/start_conversation"
    payload = {
        "thread_id": THREAD_ID,
        "doctor_name": "Dr. Sarah Johnson",
        "clinic_name": "Metro Allergy Clinic",
        "specialty": "paediatrics",
        "appointment_data": FOLLOWUP_APPOINTMENT_DATA
    }
    
    print("🚀 Starting followup bot conversation...")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print("✅ Conversation started successfully!")
        print(f"Thread ID: {THREAD_ID}")
        return True
    else:
        print(f"❌ Failed to start conversation: {response.json()}")
        return False

def send_message(message):
    """Send a message to the followup bot."""
    url = f"{BASE_URL}/message"
    payload = {
        "thread_id": THREAD_ID,
        "message": message,
        "appointment_data": FOLLOWUP_APPOINTMENT_DATA
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        bot_reply = data.get('reply', 'No reply received')
        
        # Show bot selection if it's the first message
        if 'bot_selection' in data:
            print(f"\n🤖 {data['bot_selection']}")
        
        print(f"\n🏥 Bot: {bot_reply}")
        return True
    else:
        print(f"❌ Error: {response.json()}")
        return False

def print_followup_context():
    """Print context about the followup scenario."""
    print("\n" + "="*60)
    print("📋 FOLLOWUP BOT TEST SCENARIO")
    print("="*60)
    print("Patient: You")
    print("Doctor: Dr. Sarah Johnson")
    print("Previous Appointment: 5 days ago")
    print("Reason: Seasonal allergies and asthma symptoms")
    print("Prescription:")
    print("  • Albuterol inhaler (2 puffs as needed)")
    print("  • Claritin 10mg daily") 
    print("  • Flonase nasal spray (2 sprays each nostril daily)")
    print("\nThis scenario should trigger the FOLLOWUP BOT.")
    print("="*60)

def show_sample_interactions():
    """Show sample interactions users can try."""
    print("\n💡 SAMPLE INTERACTIONS TO TRY:")
    print("-" * 40)
    print("1. 'Hi, I'm here for my follow-up appointment'")
    print("2. 'Yes, I've been taking all my medications as prescribed'")
    print("3. 'No side effects from the medications'")
    print("4. 'My symptoms have improved a lot'")
    print("5. 'I understand how to use the inhaler properly'")
    print("6. 'Actually, I've been having some new symptoms'")
    print("7. 'I forgot to take my Claritin yesterday'")
    print("8. 'The Flonase makes my nose feel weird'")
    print("-" * 40)

def interactive_chat():
    """Run the interactive chat interface."""
    print("💬 Interactive Followup Bot Chat")
    print("Type 'quit' to exit, 'help' for sample interactions")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                show_sample_interactions()
                continue
            elif not user_input:
                print("Please enter a message or 'quit' to exit.")
                continue
            
            # Send message to bot
            if not send_message(user_input):
                print("Failed to send message. Check if the server is running.")
                break
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main function to run the interactive followup bot test."""
    print("🤖 Interactive Followup Bot Tester")
    print("Make sure the Flask app is running on port 5007")
    
    # Test server connectivity
    try:
        response = requests.get(f"{BASE_URL}/")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure Flask app is running!")
        print("Run: python app.py")
        return
    
    # Show the scenario context
    print_followup_context()
    
    # Start the conversation
    if start_followup_conversation():
        show_sample_interactions()
        
        # Run interactive chat
        interactive_chat()
    else:
        print("Failed to start conversation. Check server logs.")

if __name__ == "__main__":
    main()