# Followup Bot Testing Guide

This guide provides comprehensive instructions for testing the followup bot functionality in your medical chatbot system.

## Overview

The followup bot is designed to conduct post-appointment follow-up conversations with patients. It focuses on:

1. **Medication adherence** - Checking if patients are taking prescribed medications
2. **Side effects** - Monitoring for any adverse reactions
3. **Symptom changes** - Tracking improvement or worsening of symptoms
4. **Understanding of instructions** - Ensuring patients understand their treatment plan

## When the Followup Bot is Triggered

The followup bot is automatically triggered when:
- There is a **completed appointment** with the specified doctor
- There is **NO future appointment** scheduled with the same doctor within 48 hours
- The routing logic determines this is a followup scenario

## Prerequisites

1. **Environment Setup**: Ensure you have all dependencies installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**: Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   FLASK_RUN_PORT=5007
   ```

3. **Start the Flask App**:
   ```bash
   python app.py
   ```
   The app should be running on `http://localhost:5007`

## Testing Options

### Option 1: Automated Tests (Recommended)

Run the comprehensive automated test suite:

```bash
python test_followup_bot.py
```

**What it tests:**
- ✅ Followup bot triggering with past appointments
- ✅ Followup bot NOT triggering when future appointments exist
- ✅ Complete conversation flow (medication adherence, side effects, symptoms, understanding)
- ✅ Prescription data handling
- ✅ Error scenarios

### Option 2: Interactive Testing

Run the interactive chat interface:

```bash
python interactive_followup_test.py
```

**Features:**
- 🎯 Pre-configured scenario that triggers the followup bot
- 💬 Interactive chat interface
- 💡 Built-in sample interactions to try
- 🤖 Real-time bot responses

## Test Scenarios

### Scenario 1: Successful Followup Bot Trigger

**Setup:**
- Past appointment: 5-7 days ago, status = "completed"
- No future appointments scheduled
- Doctor: "Dr. Sarah Johnson"

**Expected Result:** Follow-up Bot should be selected

**Sample Message:** "Hi, I'm here for my follow-up appointment"

**Expected Bot Response:** Should start with medication adherence question

### Scenario 2: Followup Bot Should NOT Trigger

**Setup:**
- Past appointment: 7 days ago, status = "completed"
- Future appointment: within 48 hours, status = "booked"
- Same doctor

**Expected Result:** Symptom Bot or Get Info Bot should be selected instead

### Scenario 3: Medication Adherence Flow

**Test Messages:**
1. "Yes, I've been taking my medications as prescribed"
2. "No side effects so far"
3. "My symptoms have improved significantly"
4. "Yes, I understand when to use the EpiPen"

**Expected Behavior:** Bot should ask follow-up questions without repeating previous questions

### Scenario 4: Escalation Scenario

**Test Message:** "Actually, I've been experiencing some new symptoms that worry me"

**Expected Response:** Bot should suggest contacting the clinic directly

## Sample Appointment Data Structures

### Triggers Followup Bot
```json
{
  "appointments": [
    {
      "appt_id": "apt_001",
      "appt_datetime": "2024-01-15T10:00:00Z",
      "appt_status": "completed",
      "doctor_name": "Dr. Sarah Johnson",
      "symptom-summary": "Seasonal allergies",
      "prescription": "Claritin 10mg daily, Flonase nasal spray"
    }
  ]
}
```

### Does NOT Trigger Followup Bot
```json
{
  "appointments": [
    {
      "appt_id": "apt_001",
      "appt_datetime": "2024-01-15T10:00:00Z",
      "appt_status": "completed",
      "doctor_name": "Dr. Sarah Johnson",
      "symptom-summary": "Seasonal allergies"
    },
    {
      "appt_id": "apt_002",
      "appt_datetime": "2024-01-23T14:00:00Z",
      "appt_status": "booked",
      "doctor_name": "Dr. Sarah Johnson"
    }
  ]
}
```

## API Endpoints for Manual Testing

### 1. Start Conversation
```bash
curl -X POST http://localhost:5007/start_conversation \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "test_followup_123",
    "doctor_name": "Dr. Sarah Johnson",
    "clinic_name": "Metro Allergy Clinic",
    "specialty": "paediatrics",
    "appointment_data": {
      "appointments": [
        {
          "appt_id": "apt_001",
          "appt_datetime": "2024-01-15T10:00:00Z",
          "appt_status": "completed",
          "doctor_name": "Dr. Sarah Johnson",
          "symptom-summary": "Allergic reaction",
          "prescription": "EpiPen as needed, Benadryl 25mg twice daily"
        }
      ]
    }
  }'
```

### 2. Send Message
```bash
curl -X POST http://localhost:5007/message \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "test_followup_123",
    "message": "Hi, I am here for my follow-up",
    "appointment_data": {
      "appointments": [
        {
          "appt_id": "apt_001",
          "appt_datetime": "2024-01-15T10:00:00Z",
          "appt_status": "completed",
          "doctor_name": "Dr. Sarah Johnson",
          "symptom-summary": "Allergic reaction",
          "prescription": "EpiPen as needed, Benadryl 25mg twice daily"
        }
      ]
    }
  }'
```

## Troubleshooting

### Common Issues

1. **Followup bot not triggering**
   - Check appointment data format
   - Ensure doctor names match exactly
   - Verify no future appointments exist
   - Check appointment status is "completed"

2. **Bot repeating questions**
   - This should be fixed in the current implementation
   - Check conversation history is being maintained

3. **Connection errors**
   - Ensure Flask app is running on port 5007
   - Check firewall settings
   - Verify OpenAI API key is set

### Debug Information

The system logs routing decisions and bot selections. Check the Flask app console output for:
- `LLM Routing -> followup`
- `Follow-up Bot selected`
- `--- Executing Follow-up Node ---`

## Expected Followup Bot Behavior

1. **Initial Message**: Should ask about medication adherence with the specific opening phrase
2. **Follow-up Questions**: Should cover side effects, symptom changes, and understanding
3. **No Repetition**: Should not ask the same question twice
4. **Escalation**: Should recommend contacting clinic for new/concerning symptoms
5. **Context Awareness**: Should reference specific medications from prescription data

## Success Criteria

✅ **Routing Works**: Followup bot is triggered in correct scenarios  
✅ **Questions Flow**: Bot asks relevant follow-up questions in sequence  
✅ **No Repetition**: Bot doesn't repeat previously asked questions  
✅ **Prescription Awareness**: Bot references specific medications  
✅ **Escalation**: Bot appropriately escalates concerning responses  
✅ **Error Handling**: System handles edge cases gracefully

## Next Steps

After testing, you can:
1. Customize the followup questions for your specific medical specialty
2. Adjust the routing logic timing (currently 48 hours)
3. Add more sophisticated prescription parsing
4. Integrate with your actual appointment management system