# Medical Assistant Bot

A comprehensive medical assistant bot with Streamlit frontend and Flask backend, supporting symptom collection, follow-up, and dynamic information retrieval.

## Features

- **Symptom Collection Bot**: Interactive symptom assessment and collection
- **Follow-up Bot**: Post-appointment follow-up and medication adherence tracking
- **Get Info Bot**: Dynamic web scraping and RAG-based information retrieval
- **Multi-modal Interface**: Streamlit UI with real-time chat functionality

## Deployment

### Streamlit Cloud Deployment

1. **Fork/Clone this repository**
2. **Set up environment variables** in Streamlit Cloud:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `BACKEND_URL`: URL of your Flask backend (if separate deployment)

3. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set the main file path to: `streamlit_app.py`
   - Deploy!

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

3. **Run the app**:
   ```bash
   streamlit run streamlit_app.py
   ```

## Project Structure

```
Final_Bot/
├── streamlit_app.py          # Main Streamlit application
├── app.py                    # Flask backend API
├── lance_main.py            # LangGraph bot definitions
├── conversation/            # Conversation management
├── models/                  # LLM chains and prompts
├── config/                  # Configuration files
├── utils/                   # Utility functions
└── frontend/               # React frontend (deprecated)
```

## Configuration

- Update `config/constants.py` for clinic-specific information
- Modify `config/llm_config.py` for LLM settings
- Adjust prompts in `models/prompts.py` as needed

## License

MIT License 