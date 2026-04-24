import gradio as gr
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from TTS.api import TTS
import os

# --- 0. Define Supported Languages for the TTS Model ---
# XTTSv2 model supports these languages. We'll check against this list.
XTTS_SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]

# --- Load Pre-trained Models ---

print("Loading models... This may take a few minutes.")

# Define the device to run the models on (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load SpeechBrain model for Language/Accent Identification
try:
    language_id_model = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir="pretrained_models/lang-id-voxlingua107-ecapa",
        run_opts={"device": device}
    )
    print("SpeechBrain Language ID model loaded.")
except Exception as e:
    print(f"Error loading SpeechBrain model: {e}")
    language_id_model = None

# Load Coqui XTTS model for Zero-Shot Voice Cloning
try:
    tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("Coqui TTS model loaded.")
except Exception as e:
    print(f"Error loading Coqui TTS model: {e}")
    tts_model = None

print("Model loading process finished. Check for errors above.")

# --- Define Core Function ---

def detect_accent_and_speak(text_to_speak, audio_prompt):
    """
    This function takes a text string and an audio prompt, detects the language,
    and synthesizes the text in the detected language using the voice from the audio prompt.
    """
    if language_id_model is None or tts_model is None:
        return "Models are not loaded. Please check the console for errors.", None

    if not text_to_speak or audio_prompt is None:
        return "Please provide text and record your voice.", None

    sampling_rate, audio_data_numpy = audio_prompt
    audio_tensor = torch.from_numpy(audio_data_numpy).float()

    resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
    resampled_audio = resampler(audio_tensor)

    # --- Language Identification ---
    try:
        prediction = language_id_model.classify_batch(resampled_audio.unsqueeze(0))
        
        # The prediction object contains multiple pieces of info.
        # prediction[1][0] gives the language code (e.g., 'en', 'es')
        # prediction[3][0] gives the full name (e.g., 'en: English')
        detected_lang_code = prediction[1][0]
        detected_language_name = prediction[3][0].split(': ')[1]

        # Check if the detected language is supported by the TTS model
        if detected_lang_code in XTTS_SUPPORTED_LANGUAGES:
            tts_language = detected_lang_code
            detection_result = f"Detected Language: {detected_language_name} ({tts_language})"
        else:
            # If not supported, default to English and notify the user
            tts_language = "en"
            detection_result = (f"Detected {detected_language_name}, which is not supported for speech synthesis. "
                                f"Defaulting to English.")

    except Exception as e:
        detection_result = f"Could not detect language: {e}"
        return detection_result, None

    # --- Zero-Shot Voice Cloning and Synthesis ---
    try:
        prompt_path = "prompt.wav"
        torchaudio.save(prompt_path, audio_tensor.unsqueeze(0), sampling_rate)

        print(f"Synthesizing text: '{text_to_speak}' in language '{tts_language}' with the voice from '{prompt_path}'")

        output_wav_path = "output.wav"
        tts_model.tts_to_file(
            text=text_to_speak,
            speaker_wav=prompt_path,
            language=tts_language,  # Use the dynamically detected language
            file_path=output_wav_path
        )
        print("Synthesis complete.")
        return detection_result, output_wav_path
    except Exception as e:
        print(f"Error during speech synthesis: {e}")
        return f"{detection_result}\n\nSpeech Synthesis Error: {e}", None

# ---  Create the Gradio Interface ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Multilingual Accent Detector and Voice Cloner\n"
        "1. **Record your voice** in your native language (5-10 seconds is best).\n"
        "2. **Enter text in the SAME language** you just spoke.\n"
        "3. Click **Generate Speech** to hear the text spoken in your voice and language."
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Record Your Voice Here (The Language Prompt)")
            text_input = gr.Textbox(label="Text to Speak (in the same language as your recording)")
            generate_button = gr.Button("Generate Speech")
        with gr.Column():
            accent_output = gr.Label(label="Language Detection Result")
            audio_output = gr.Audio(label="Synthesized Speech Output")

    generate_button.click(
        detect_accent_and_speak,
        inputs=[text_input, audio_input],
        outputs=[accent_output, audio_output]
    )

# --- Launch the Demo ---
if __name__ == "__main__":
    demo.launch(debug=True, share=True)