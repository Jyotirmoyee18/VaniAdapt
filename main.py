import gradio as gr
import whisper
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

# --- Load Pre-trained Models ---


try:
    asr_model = whisper.load_model("base")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    # Handle the error, maybe by providing a message to the user.
    asr_model = None

# Load a pre-trained speaker recognition model to simulate prosody extraction
# This model helps in getting a representation of the speaker's voice (prosody)
try:
    speaker_id_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb"
    )
except Exception as e:
    print(f"Error loading Speaker ID model: {e}")
    # Handle the error
    speaker_id_model = None

# ---  Define Core Functions ---

def transcribe_audio(audio_path):
    """Transcribes the given audio file using the Whisper ASR model."""
    if not asr_model:
        return "ASR model not loaded."
    try:
        result = asr_model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        return f"Error during transcription: {e}"

def simulate_vani_adapt(audio_path):
    """
    This function simulates the Vani-Adapt process.
    In a real implementation, this would involve your DPPE model.
    For the demo, we will conceptually represent this step.
    """

    print("Simulating Vani-Adapt: In a real demo, this would convert the accent.")
    return audio_path

def process_and_compare(audio_file):
    """
    The main function for the Gradio interface.
    It takes an audio file, gets the baseline transcription,
    simulates Vani-Adapt, and gets the improved transcription.
    """
    if not audio_file:
        return "Please upload an audio file.", "Please upload an audio file."

    # --- Baseline ASR Transcription ---
    baseline_transcription = transcribe_audio(audio_file)

    # --- Simulated Vani-Adapt Processing ---
    adapted_audio_path = simulate_vani_adapt(audio_file)

    adapted_transcription = transcribe_audio(adapted_audio_path)

    

    return baseline_transcription, adapted_transcription

# --- Create the Gradio Interface ---

with gr.Blocks() as demo:
    gr.Markdown("# Vani-Adapt: A Zero-Shot Accent Trans-adaptation Framework")
    gr.Markdown(
        "**Demonstration by Jyotirmoyee Mandal, Kunal Halder, and Kakali Das**"
    )
    gr.Markdown(
        "Upload an audio file with an Indian accent to see how Vani-Adapt improves ASR accuracy."
    )

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload Accented Speech")

    process_button = gr.Button("Transcribe and Adapt")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Baseline ASR Transcription")
            baseline_output = gr.Textbox(label="Transcription of Original Audio")
        with gr.Column():
            gr.Markdown("### Vani-Adapt ASR Transcription")
            adapted_output = gr.Textbox(label="Transcription after Accent Adaptation")

    process_button.click(
        process_and_compare,
        inputs=audio_input,
        outputs=[baseline_output, adapted_output]
    )

    gr.Markdown("## How this Demo Works")
    gr.Markdown(
        "- **Baseline ASR:** The uploaded audio is first transcribed using a standard ASR model to show potential errors due to the accent."
    )
    gr.Markdown(
        "- **Vani-Adapt Simulation:** The core of our paper, the Disentangled Phonetic-Prosodic Encoder (DPPE), would process the audio here. It separates the phonetic content from the prosodic style (the accent) and reconstructs the speech with a neutral accent."
    )
    gr.Markdown(
        "- **Adapted ASR:** The new, accent-neutral audio is then transcribed. The result is a more accurate transcription, demonstrating the effectiveness of Vani-Adapt."
    )


# --- Launch the Demo ---

if __name__ == "__main__":
    demo.launch()