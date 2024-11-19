def detect_speech_activity(audio_data) -> bool:
    """Detect speech activity based on audio data and returns True"""
    pass

def transcribe_audio(audio_data) -> str:
    """Transcribe audio data using Faster Whisper."""
    return whisper_model.transcribe()        

def main():    
    # 1. Keep listening to audio stream, never stop this process
    interviewer_audio_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback)

    detect_speech_activity(interviewer_audio_stream)

    # 2. Look for speech activity
    # 3. If Speech activity detected, start transcribing audio
    # 4. If Speech activity ends (silence for 2 seconds) or receives user input like "Enter", collect the completed transcribed question and make get_answer_request..
    # 5. get_answer_request must be made in a new async thread and start streaming the response...
    # 6. If new speech activity detected, start transcribing audio, and repeat from step 4