import gradio as gr
import torch
from TTS.api import TTS
import whisper 
import os
import torchaudio

# --- Load Pre-trained Models ---

print("Loading models... This may take a few minutes on first run.")

# Define the device to run the models on (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Bengali TTS Model
try:
    print("Loading Bengali TTS model...")
    tts_model = TTS("tts_models/indic/vits-v1/bn/vits").to(device)
    print("Bengali TTS model loaded.")
except Exception as e:
    print(f"Error loading Coqui TTS model: {e}")
    tts_model = None

# Load Whisper Speech-to-Text Model
try:
    print("Loading Whisper STT model...")
    # "base" is a small and fast model. For higher accuracy, you can use "medium" or "large".
    whisper_model = whisper.load_model("base", device=device)
    print("Whisper STT model loaded.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    whisper_model = None

print("Model loading process finished.")

# --- Define Core Function ---

def transcribe_and_respond(audio_prompt):
    """
    This function takes an audio input, transcribes it to text,
    extracts a name, generates a response in Bengali, and synthesizes it to speech.
    """
    # Check if models were loaded successfully
    if tts_model is None or whisper_model is None:
        error_msg = "One or more models are not loaded. Please check the console for errors."
        return error_msg, None

    # Check for valid input
    if audio_prompt is None:
        return "Please record your voice first.", None

    # Unpack audio data from Gradio input
    sampling_rate, audio_data_numpy = audio_prompt
    
    # Save the recorded audio to a temporary file for Whisper to process
    input_audio_path = "input_audio.wav"
    torchaudio.save(input_audio_path, torch.from_numpy(audio_data_numpy).float().unsqueeze(0), sampling_rate)

    # --- Transcribe Speech to Text ---
    try:
        print("Transcribing audio...")
        # Transcribe the audio file
        result = whisper_model.transcribe(input_audio_path, fp16=False if device == "cpu" else True)
        transcribed_text = result['text'].strip()
        print(f"Transcribed text: {transcribed_text}")
    except Exception as e:
        print(f"Error during transcription: {e}")
        return f"Could not transcribe audio: {e}", None

    # --- Extract Name and Generate Response ---
    name = "বন্ধু" # Default name is "friend" in Bengali
    # Simple logic to find the name after "myself"
    if "myself" in transcribed_text.lower():
        # Split the string at "myself" and take the part after it
        try:
            name = transcribed_text.lower().split("myself")[1].strip()
            # Capitalize the first letter for better display
            name = name.capitalize()
        except IndexError:
            # This happens if "myself" is the last word
            pass
    
    # Create the response in Bengali
    response_text = f"হ্যালো {name}, কেমন আছো?"
    print(f"Generated response: {response_text}")

    # ---Synthesize Response to Speech ---
    try:
        output_wav_path = "response_bengali.wav"
        tts_model.tts_to_file(
            text=response_text,
            file_path=output_wav_path
        )
        print("Synthesis complete.")
        # Return the transcribed text (to show in the UI) and the path to the output audio
        return f"Heard: '{transcribed_text}'", output_wav_path
    except Exception as e:
        print(f"Error during speech synthesis: {e}")
        return f"Heard: '{transcribed_text}'.\nSynthesis Error: {e}", None

# --- Create the Gradio Interface ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Conversational AI (English to Bengali)\n"
        "1. **Speak in English** into the microphone (e.g., 'Hello myself John Doe').\n"
        "2. Click **Submit** and wait for the Bengali audio response."
    )

    with gr.Row():
        with gr.Column():
            # Changed from Textbox to Audio input
            audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Speak Here")
            submit_button = gr.Button("Submit")
        with gr.Column():
            transcription_output = gr.Label(label="Transcription Result")
            audio_output = gr.Audio(label="Bengali Response")

    submit_button.click(
        transcribe_and_respond,
        inputs=[audio_input],
        outputs=[transcription_output, audio_output]
    )

# ---  Launch the Demo ---
if __name__ == "__main__":
    demo.launch(debug=True, share=True)