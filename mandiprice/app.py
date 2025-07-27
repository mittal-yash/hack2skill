import os
import openai
import gradio as gr
import utils  # make sure this exists or replace with placeholder HTML

# Set your OpenAI API key (alternatively use ENV var)
openai.api_key = os.getenv("OPENAI_API_KEY", "API_KEY")

# ------------------------------
# Function: Transcribe audio using OpenAI Whisper
def transcribe_audio(audio_file_path: str) -> str:
    with open(audio_file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]

# ------------------------------
# Function: Generate speech from text using OpenAI TTS
def synthesize_speech(text: str, output_filename: str = "/tmp/response.mp3") -> str:
    response = openai.audio.speech.create(
        model="tts-1",
        voice="shimmer",  # shimmer | alloy | echo | fable | nova | onyx
        input=text
    )
    with open(output_filename, "wb") as out:
        out.write(response.read())
    return output_filename

# ------------------------------
# Function: Orchestrate the voice flow
def generate_with_voice(audio, history: list[gr.ChatMessage], request: gr.Request):
    if audio is None:
        yield "Please provide an audio input."
        return

    # Step 1: Transcribe audio to text
    user_text = transcribe_audio(audio)
    if not user_text:
        yield "Sorry, I could not understand the audio. Please try again."
        return

    # Step 2: Call your existing chatbot logic (stub here)
    # Replace `generate` with your chatbot logic that yields messages
    responses = []
    for output in generate(user_text, history, request):  # your chatbot logic
        responses = output

    # Step 3: Convert chatbot text to speech
    combined_response = " ".join(part if isinstance(part, str) else "" for part in responses)
    audio_response_path = synthesize_speech(combined_response)
    yield audio_response_path

# ------------------------------
# Placeholder chatbot logic (to be replaced with your logic)
def generate(user_text, history, request):
    # Replace with your model call
    bot_reply = f"You said: {user_text}"
    yield [bot_reply]

# ------------------------------
# Gradio UI setup
with gr.Blocks(theme=utils.custom_theme if hasattr(utils, "custom_theme") else None) as demo:
    with gr.Row():
        gr.HTML(utils.public_access_warning if hasattr(utils, "public_access_warning") else "<h3>Voice Assistant</h3>")

    with gr.Row():
        with gr.Column(scale=2, variant="panel"):
            audio_in = gr.Audio(source="upload", type="filepath", label="Record or Upload Audio")
            audio_out = gr.Audio(label="Bot's Voice Response")
            btn = gr.Button("Ask via Voice")
            btn.click(fn=generate_with_voice, inputs=[audio_in, gr.State([]), gr.Request()], outputs=audio_out)

    with gr.Row():
        gr.HTML("""
            <div style="text-align:center; padding: 10px; font-size: 14px; color: #666;">
            Built with ❤️ using OpenAI + Gradio
            </div>
        """)

demo.launch(server_name="0.0.0.0", server_port=8080, show_error=True)
