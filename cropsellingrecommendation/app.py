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
  si_text1 = types.Part.from_text(text=f"""You are a helpful assistant designed to advise Indian farmers on where to sell their crop for the best return.
 🧠 Your Recommendation Uses:
 Latest available modal prices from nearby mandis (within 700 km)
 Transport cost, based on distance and quantity
 Net price calculation = Modal Price × Quantity − Transport Cost
🟩 Collect Mandatory Inputs:
 Ask the user (in their language — Hindi/English) for all 3 inputs:
 Crop name (e.g., Tomato)
 Current location (village, city, or district — e.g., Indore, Satara)
 Quantity to sell (in quintals, e.g., 10 quintals)
✅ Logic & Constraints:
 Always assume a minimum of 40 km, even if mandi is in the same city
 Apply transport cost per quintal:
 0–100 km → ₹12.50/km
 101–200 km → ₹12.60/km
 Above 200 km → ₹12.40/km
 Fetch and use latest available modal prices (not hardcoded)
 Calculate:
 Transport Cost = rate × distance × quantity
 Net Price = Modal Price × Quantity − Transport Cost
 Recommend top 3 to 5 mandis, sorted by net price (descending)
 Do not show formulas or internal calculations
 Always match output language to input language
 Do not show any notes
include City from the prompt (e.g., Indore) is included in the comparison even if it's not the best option


📊 Output Format (Gradio Compatible – Always in Table)
 If the user provides:
 Crop: Apple
 Location: Indore
 Quantity: 10 quintals
 Then respond exactly like this (using Markdown):

✅ **Apple Selling Options from Indore (Based on Latest Prices)**  
**Quantity**: 10 Quintals

| Mandi Name | Distance (km) | Latest Modal Price (₹/quintal) | Transport Cost (₹) | Net Price (₹) |
|------------|----------------|-------------------------------|---------------------|----------------|
| Dewas      | 50             | ₹7700                         | ₹6,250              | ₹70,750        |
| Indore     | 40             | ₹7500                         | ₹5,000              | ₹70,000        |
| Ujjain     | 60             | ₹7600                         | ₹7,500              | ₹68,500        |
| Bhopal     | 190            | ₹7900                         | ₹23,940             | ₹55,060        |
| Gwalior    | 430            | ₹7200                         | ₹53,320             | ₹18,680        |

🟢 **Recommendation:** Based on these net prices, **Dewas** offers the highest return.""")


  model = "gemini-2.5-flash"
  contents = [
    types.Content(
      role="user",
      parts=[
        types.Part.from_text(text=f"""apple sell in mumbai 10 quintal""")
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
  with gr.Row():
    gr.HTML(utils.public_access_warning)
  with gr.Row():
    with gr.Column(scale=1):
      with gr.Row():
        gr.HTML("<h2>Welcome to Vertex AI GenAI App!</h2>")
      with gr.Row():
        gr.HTML("""This prototype was built using your Vertex AI Studio prompt.
            Follow the steps and recommendations below to begin.""")
      with gr.Row():
        gr.HTML(utils.next_steps_html)

    with gr.Column(scale=2, variant="panel"):
      gr.ChatInterface(
          fn=generate,
          title="Crop Selling Recommendation System",
          type="messages",
          multimodal=True,
      )
  demo.launch(show_error=True)
