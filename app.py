import os
import gradio as gr
from huggingface_hub import login
from pipe import AudioSpeechNERPipeline
import html

# Optimized Labels Dictionary
LABELS = {
    0: 'O', 1: 'B-DATE', 2: 'B-EVENT', 3: 'B-LOC', 
    4: 'B-ORG', 5: 'B-PER', 6: 'I-DATE', 7: 'I-EVENT', 
    8: 'I-LOC', 9: 'I-ORG', 10: 'I-PER'
}

def process_audio_pipeline(audio):
    """Robust Gradio processing function"""
    pipeline = AudioSpeechNERPipeline()
    
    try:
        transcription, entities = pipeline.process_audio(audio)
        highlighted_text = highlight_entities(transcription, entities)
        
        return transcription, highlighted_text
    
    except Exception as e:
        return f"Error processing audio: {str(e)}", ""

def highlight_entities(transcription, entities):
    """Enhanced entity highlighting with a legend."""
    # Map entity labels to human-readable labels if needed
    processed_entities = [
        {**entity, 'label': LABELS[int(entity['entity'].split("_")[-1])]}
        for entity in entities if int(entity['entity'].split("_")[-1]) != 0
    ]
    
    # Sort entities by their start position to avoid overlapping issues
    processed_entities.sort(key=lambda x: x.get('start', 0))
    
    # Escape transcription for HTML safety
    transcription = html.escape(transcription)
    highlighted_text = transcription
    offset = 0  # Track how much the text length changes due to added HTML
    
    # Define color coding for entity types
    colors = {
        'B-PER': 'blue', 'I-PER': 'blue',
        'B-ORG': 'green', 'I-ORG': 'green',
        'B-LOC': 'red', 'I-LOC': 'red',
        'B-DATE': 'purple', 'I-DATE': 'purple',
        'B-EVENT': 'orange', 'I-EVENT': 'orange'
    }
    
    for entity in processed_entities:
        start = entity.get('start', 0) + offset
        end = entity.get('end', 0) + offset
        label = entity['label']
        
        color = colors.get(label, 'black')
        
        # Wrap the entity text with a styled span
        highlighted_part = (
            f'<span style="background-color: {color}; color: white; '
            f'padding: 2px; border-radius: 3px;">'
            f'{highlighted_text[start:end]}</span>'
        )
        
        # Replace text in the highlighted_text with the HTML
        highlighted_text = (
            highlighted_text[:start] + highlighted_part +
            highlighted_text[end:]
        )
        
        # Update offset to account for added HTML
        offset += len(highlighted_part) - (end - start)
    
    # Create a legend for the labels and their colors
    legend = '<br><br><strong>Legend:</strong><br>'
    legend += ''.join(
        f'<span style="background-color: {color}; color: white; '
        f'padding: 2px; border-radius: 3px; margin-right: 10px;">{label}</span>'
        for label, color in colors.items()
    )
    
    return highlighted_text + legend


def create_gradio_interface():
    """Enhanced Gradio interface with improved styling"""
    iface = gr.Interface(
        fn=process_audio_pipeline,
        inputs=gr.Audio(type="filepath", label="Upload Uzbek Audio"),
        outputs=[
            gr.Textbox(label="Transcription"),
            gr.HTML(label="Named Entities")  # Changed to HTML for highlighting
        ],
        title="🎙️ Uzbek Speech Recognition & NER",
        description=(
            "Upload an Uzbek audio file to transcribe and "
            "visualize named entities with color-coded highlighting. "
            "Supports MP3 and WAV formats."
        ),
        css=".gradio-container { background-color: #f0f0f0; }"
    )
    return iface

def main():
    """Main execution function"""
    demo = create_gradio_interface()
    demo.launch()

if __name__ == "__main__":
    # Optional: Handle HuggingFace login more securely
    token = os.getenv('HF_TOKEN')
    if token:
        login(token=token, new_session=False)
    
    main()
