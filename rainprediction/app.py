"""Main entry point for the app.

This app is generated based on your prompt in Vertex AI Studio using
Google GenAI Python SDK (https://googleapis.github.io/python-genai/) and
Gradio (https://www.gradio.app/).

You can customize the app by editing the code in Cloud Run source code editor.
You can also update the prompt in Vertex AI Studio and redeploy it.
"""

import base64
from google import genai
from google.genai import types
import gradio as gr
import utils


def generate(
    message,
    history: list[gr.ChatMessage],
    request: gr.Request
):
  """Function to call the model based on the request."""

  validate_key_result = utils.validate_key(request)
  if validate_key_result is not None:
    yield validate_key_result
    return

  client = genai.Client(
      vertexai=True,
      project="kissan-465113",
      location="global",
  )
  msg2_text1 = types.Part.from_text(text=f"""Mumbai is predicted to experience light to heavy rain for the next week. There is a high chance of rain, with some days having \"rather heavy rain\" or \"heavy rain\". Temperatures will generally be around 26-28°C (80-84°F).""")


  model = "gemini-2.5-flash"
  contents = [
    types.Content(
      role="user",
      parts=[
        types.Part.from_text(text=f"""rain prediction in mumbai""")
      ]
    ),
    types.Content(
      role="model",
      parts=[
        msg2_text1
      ]
    ),
    types.Content(
      role="user",
      parts=[
        types.Part.from_text(text=f"""rain prediction in mumbai""")
      ]
    ),
  ]

  for prev_msg in history:
    role = "user" if prev_msg["role"] == "user" else "model"
    parts = utils.get_parts_from_message(prev_msg["content"])
    if parts:
      contents.append(types.Content(role=role, parts=parts))

  if message:
    contents.append(
        types.Content(role="user", parts=utils.get_parts_from_message(message))
    )

  tools = [
      types.Tool(google_search=types.GoogleSearch()),
  ]
  generate_content_config = types.GenerateContentConfig(
      temperature=0.9,
      top_p=0.95,
      seed=0,
      max_output_tokens=65535,
      safety_settings=[
          types.SafetySetting(
              category="HARM_CATEGORY_HATE_SPEECH",
              threshold="OFF"
          ),
          types.SafetySetting(
              category="HARM_CATEGORY_DANGEROUS_CONTENT",
              threshold="OFF"
          ),
          types.SafetySetting(
              category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
              threshold="OFF"
          ),
          types.SafetySetting(
              category="HARM_CATEGORY_HARASSMENT",
              threshold="OFF"
          )
      ],
      tools=tools,
      system_instruction=[types.Part.from_text(text=f"""Rain prediction for 1 week for given location give one line answer. Also mention date range""")],
  )

  results = []
  for chunk in client.models.generate_content_stream(
      model=model,
      contents=contents,
      config=generate_content_config,
  ):
    if chunk.candidates and chunk.candidates[0] and chunk.candidates[0].content:
      results.extend(
          utils.convert_content_to_gr_type(chunk.candidates[0].content)
      )
      if results:
        yield results

with gr.Blocks(theme=utils.custom_theme) as demo:
    gr.HTML(
    """
    <style>
        footer { display: none !important; }
    </style>
    """
    )

    with gr.Row():
        with gr.Column(scale=2, variant="panel"):
            gr.ChatInterface(
                fn=generate,
                title="Rain Prediction",
                type="messages",
                multimodal=True,
                examples=["Say Hi! to start"]
            )

    gr.HTML(
    """
    <div style="text-align: center; font-size: 14px; padding: 1em; color: #555;">
        Built with ❤️ by <strong>Saygen.ai</strong>
    </div>
    """
    )


demo.launch(show_error=True, share=False, show_api=False)
