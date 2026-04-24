
print("--- TESTING WHISPER MODEL ---")
try:
    import whisper
    # Using "tiny" model for the smallest possible download and memory usage
    whisper_model = whisper.load_model("tiny")
    print(" Whisper Model loaded successfully!")
except Exception as e:
    print(" FAILED to load Whisper model.")
    # This will print the specific error for Whisper
    print(e)

print("\n" + "="*50 + "\n")

print("--- TESTING TTS MODEL ---")
try:
    from TTS.api import TTS
    tts_model = TTS("tts_models/indic/vits-v1/bn/vits")
    print(" TTS Model loaded successfully!")
except Exception as e:
    print(" FAILED to load TTS model.")
    # This will print the specific error for TTS
    print(e)