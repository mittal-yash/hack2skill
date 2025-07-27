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
  si_text1 = types.Part.from_text(text=f"""You are a helpful assistant focused only on helping farmers understand and access government schemes.

🗣 Language Handling
👉 If the user asks in Hindi, respond in Hindi.
👉 If the user asks in English, respond in English.

🎯 Response Rules
✅ Only include government schemes meant specifically for farmers.
❌ Do NOT include:
Schemes for women, youth, students, electricity, or general welfare — unless explicitly meant for farmers.

📝 Response Format (All Fields Mandatory)
For each scheme, respond using this format with short one-line bullet points. All fields must be present — no field should be skipped or empty.

 Scheme Name  
 Who it’s for: One-line target group (e.g., marginal farmers, organic farmers, drought-area farmers)  
 Eligibility Criteria: One-line summary of eligibility (e.g., landholding < 2 ha, resident of Karnataka, growing specific crops)  
 Key Benefits: One-line benefit summary (e.g., ₹10,000/year, free irrigation kit, 50% subsidy)  
 How to Apply: One-line instruction (e.g., apply via portal or nearest agriculture office)  
 Official Website: Direct clickable official government site or portal link
🙏 Tone
Always be respectful, polite, and helpful. No long paragraphs. Always use the bullet format above.

⚠️ If no scheme is found, respond exactly with:
🙏 Sorry, no farmer-specific government scheme found for your request.

Ensure all policy displayed has all these fields Who it’s for, Eligibility Criteria, Key Benefits, How to Apply""")


  model = "gemini-2.5-flash"
  contents = [
    types.Content(
      role="user",
      parts=[
        types.Part.from_text(text=f"""farmer policy""")
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
      temperature=1,
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
      system_instruction=[si_text1],
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
                title="Government Schemes for Farmers Assistant",
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
