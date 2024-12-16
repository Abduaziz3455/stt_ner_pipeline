# Gradio App

This repository hosts a Gradio-based application for audio processing that performs Speech-to-Text (STT) and Named Entity Recognition (NER).

## Features
1. **Speech-to-Text (STT):** Converts audio files into transcribed text.
2. **Named Entity Recognition (NER):** Extracts named entities (e.g., names, locations, organizations) from transcribed text.
3. **Gradio UI:** An easy-to-use interface powered by Gradio for file uploads and results visualization.

## Prerequisites
Ensure you have the following installed:
- Python 3.10 or later
- pip (Python package manager)

## Installation
Follow these steps to set up and run the application:

1. **Clone the repository**
   ```bash
   git clone https://github.com/Abduaziz3455/stt_ner_pipeline.git
   cd gradio_app
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Start the Gradio app**:
   ```bash
   python app.py
   ```
   
   - The application will start, and a URL will be generated, such as `http://127.0.0.1:7860`.
   - Open this link in your browser.

2. **Upload your audio file**:
   - On the first upload, the necessary models (STT and NER) will be automatically downloaded.
   - Wait for the models to load (this happens only once).
   - Once ready, you can upload audio files to get transcriptions and named entities.

## File Structure
```
 gradio_app/
 ├── .gradio            # Gradio-specific configurations
 ├── .env               # Environment variables (optional)
 ├── .gitignore         # Git ignore file
 ├── app.py             # Main Gradio application
 ├── pipe.py            # Processing pipeline (STT & NER logic)
 ├── requirements.txt   # Dependencies list
 ├── README.md          # Documentation (you are here)
 ├── fine_tune_whisper.ipynb  # Fine-tuning Whisper notebook
 ├── ner_roberta_uz.ipynb     # Named Entity Recognition notebook
 ├── documentation.docx       # Additional documentation
 ├── presentation.pptx       # Project presentation slides
```

## Example Usage
1. Run the app using `python app.py`.
2. Open the generated Gradio link in your browser.
3. Upload an audio file (e.g., `audio_sample.wav`).
4. View the transcription and extracted named entities.

### Sample Screenshot:
![Gradio App Screenshot](https://drive.google.com/file/d/1-6phuLv-ryM-5auMGfQEAkJiBV5dnSuM/view)

## Troubleshooting
- **Models not downloading:** Ensure your internet connection is active.
- **Dependencies issues:** Re-run `pip install -r requirements.txt` to ensure all dependencies are installed.
- **Gradio UI not loading:** Restart the app using `python app.py`.

---
Feel free to contact me for questions or feedback! abduaziz3455@gmail.com
