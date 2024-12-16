import torch
import librosa
import noisereduce as nr
import numpy as np
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer, AutoTokenizer

class AudioSpeechNERPipeline:
    def __init__(
        self, 
        stt_model_name='abduaziz/whisper-small-uzbek', 
        ner_model_name='abduaziz/roberta-ner-uzbek', 
        stt_language='uz',
        chunk_duration=30
    ):
        # Use lazy loading for pipelines
        self.stt_pipeline = None
        self.ner_pipeline = None
        self.stt_model_name = stt_model_name
        self.ner_model_name = ner_model_name
        self.chunk_duration = chunk_duration

    def load_whisper_model(self, model_name='abduaziz/whisper-small-uzbek'):
        try:
            # Load processor
            processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Uzbek", task="transcribe")
            
            # Load model
            model = WhisperForConditionalGeneration.from_pretrained(model_name)
            
            return model, processor
        
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            raise

    def _load_pipelines(self):
        """Lazy load pipelines only when needed"""
        if self.stt_pipeline is None:
            # Load Whisper model and processor explicitly
            model, processor = self.load_whisper_model(self.stt_model_name)
            tokenizer = AutoTokenizer.from_pretrained('abduaziz/whisper-small-uzbek')
            self.stt_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                processor=processor,
                feature_extractor = processor.feature_extractor,
                tokenizer=tokenizer,
                return_timestamps=True
            )
        if self.ner_pipeline is None:
            self.ner_pipeline = pipeline(
                task="ner",
                model=self.ner_model_name
            )

    def chunk_audio(self, audio, sample_rate):
        """More efficient audio chunking"""
        chunk_samples = self.chunk_duration * sample_rate
        return [
            {'array': audio[start:start+chunk_samples], 'sampling_rate': sample_rate}
            for start in range(0, len(audio), chunk_samples)
        ]

    def transcribe_audio(self, audio_path):
        """Enhanced audio transcription with better error handling"""
        self._load_pipelines()
        
        audio, sample_rate = librosa.load(audio_path, sr=16000)
        preprocessed_audio = preprocess_audio(audio, sr=sample_rate)
        
        if preprocessed_audio is None:
            raise ValueError("Audio preprocessing failed")

        if len(preprocessed_audio) / sample_rate > self.chunk_duration:
            chunks = self.chunk_audio(preprocessed_audio, sample_rate)
            transcriptions = [
                self.stt_pipeline(chunk)['text'] for chunk in chunks
            ]
            return " ".join(transcriptions)
        
        return self.stt_pipeline({
            'array': preprocessed_audio,
            'sampling_rate': sample_rate
        })['text']

    def process_audio(self, audio_path):
        """Streamlined audio processing"""
        transcription = self.transcribe_audio(audio_path)
        
        self._load_pipelines()
        entities = self.ner_pipeline(transcription)
        
        return transcription, entities

def preprocess_audio(audio_array, sr=16000):
    """Improved audio preprocessing with better type handling"""
    try:
        # Handle tensor or numpy array input
        if isinstance(audio_array, torch.Tensor):
            audio_array = audio_array.numpy()
        
        # Convert stereo to mono
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=0)
        
        # Noise reduction and normalization
        noise_reduced = nr.reduce_noise(
            y=audio_array, 
            sr=sr, 
            prop_decrease=0.5,
            n_std_thresh_stationary=1.5
        )
        
        normalized_audio = librosa.util.normalize(noise_reduced)
        trimmed_audio, _ = librosa.effects.trim(normalized_audio, top_db=25)
        
        return trimmed_audio.astype(np.float32)
    
    except Exception as e:
        print(f"Audio preprocessing error: {e}")
        return None
