import sounddevice as sd
import webrtcvad
import numpy as np
import queue
import time
from scipy.signal import resample
from faster_whisper import WhisperModel
from openai import OpenAI

# Configuration
DEVICE = "cuda"  # Use GPU for Faster Whisper
MODEL_PATH = "medium"  # Choose model size for Whisper
SAMPLE_RATE = 16000  # Whisper model expects 16 kHz
THRESHOLD = 0.01  # Silence threshold for speech detection
SILENCE_DURATION = 2  # Silence duration in seconds to detect end of speech

# Initialize Faster Whisper
model = WhisperModel(MODEL_PATH, device=DEVICE, compute_type="int8_float16")

# LLM Configuration
from config.secrets import llm
client = OpenAI(base_url=llm["base_url"], api_key=llm["api_key"])

# Buffer to capture audio in real-time
audio_buffer = queue.Queue()


def select_device(device_name="Stereo Mix Recording (Realtek(R) Audio)", host_api_name="Windows WASAPI"):
    for device in sd.query_devices():
        if device_name in device["name"] and host_api_name in sd.query_hostapis()[device["hostapi"]]["name"]:
            print(f"Selected device index: {device["index"]}, device name: {device_name}, host API name: {host_api_name}")
            return device["index"], device["default_samplerate"]
    default_device_index = sd.default.device[0]
    return default_device_index, sd.query_devices(default_device_index)["default_samplerate"]


def audio_callback(inData, frames, time_info, status):
    """Audio callback to capture audio data."""
    if status:
        print(status)
    audio_buffer.put(inData.copy())


def resample_audio(audio_data, original_rate, target_rate=16000):
    num_samples = int(len(audio_data) * target_rate / original_rate)
    return resample(audio_data, num_samples)


import webrtcvad
import numpy as np

def is_speech_activity(audio_data, sample_rate=48000, frame_duration_ms=20):
    """
    Detect speech activity using WebRTC VAD.
    :param audio_data: Raw audio data (float32 or PCM 16-bit).
    :param sample_rate: The sample rate of the audio data (must be an integer, 8000, 16000, 32000, or 48000 Hz).
    :param frame_duration_ms: Duration of each frame in milliseconds (10, 20, or 30 ms).
    :return: True if speech is detected, False otherwise.
    """
    vad = webrtcvad.Vad()
    vad.set_mode(3)  # Most aggressive mode for VAD

    # Ensure the sample rate is supported
    if sample_rate not in [8000, 16000, 32000, 48000]:
        raise ValueError("Sample rate must be 8000, 16000, 32000, or 48000 Hz.")
    sample_rate = int(sample_rate)  # Ensure the sample rate is an integer

    # Convert float32 audio data to 16-bit PCM
    if audio_data.dtype != np.int16:
        audio_data = (audio_data * 32767).astype(np.int16)

    # Convert audio to bytes for VAD processing
    audio_bytes = audio_data.tobytes()

    # Calculate frame size
    frame_length = int(sample_rate * frame_duration_ms / 1000) * 2  # Frame size in bytes (16-bit PCM)

    # Process each frame
    for start in range(0, len(audio_bytes), frame_length):
        frame = audio_bytes[start:start + frame_length]
        if len(frame) < frame_length:
            break  # Ignore incomplete frames
        if vad.is_speech(frame, sample_rate):
            return True
    return False



def transcribe_audio(audio_data):
    """Transcribe audio data using Faster Whisper."""
    transcriptions, info = model.transcribe(audio_data, beam_size=5)
    return transcriptions, info

    

def generate_openai_response(history: list[dict]):
    """Generate a response using OpenAI Chat Completion API."""
    response = client.chat.completions.create(
        model=llm["default_model"],
        messages=history,
        stream=True
    )
    return response


def main():
    """Main application logic."""
    print("Starting Interview Assistant...")
    INPUT_DEVICE, INPUT_SAMPLE_RATE = select_device()
    with sd.InputStream(samplerate=INPUT_SAMPLE_RATE, channels=1, callback=audio_callback, device=INPUT_DEVICE):
        speech_active = False
        silence_start = None
        history = []

        while True:
            audio_data = []
            while not audio_buffer.empty():
                audio_data.append(audio_buffer.get())  # resample_audio(audio_buffer.get(), INPUT_SAMPLE_RATE))

            if audio_data:
                audio_data = np.concatenate(audio_data, axis=0).flatten()
                if is_speech_activity(audio_data, INPUT_SAMPLE_RATE):
                    speech_active = True
                    silence_start = None
                elif speech_active and (silence_start is None):
                    silence_start = time.time()

                if silence_start and (time.time() - silence_start > SILENCE_DURATION):
                    speech_active = False
                    silence_start = None

                    # Transcribe the question
                    print("Processing question...")
                    transcriptions, info = transcribe_audio(audio_data)
                    print(f"Detected language '{info.language}' with probability {info.language_probability}")
                    question = ""
                    for segment in transcriptions:
                        print(f"[{segment.start}s -> {segment.end}s] {segment.text}")
                    print(f"Interviewer: {question}")

                    # Add to chat history
                    history.append({"role": "user", "content": question})

                    # Generate ideal response
                    print("Generating response...")
                    response_stream = generate_openai_response(history)

                    answer = ""
                    print("Response:")
                    for chunk in response_stream:
                        if "content" in chunk.choices[0].delta:
                            chunkMessage = chunk.choices[0].delta.content
                            answer += chunkMessage
                            print(chunkMessage, end="", flush=True)

                    print("\n")
                    history.append({"role": "assistant", "content": answer})

            


if __name__ == "__main__":
    main()
