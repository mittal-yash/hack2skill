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
  si_text1 = types.Part.from_text(text=f"""You are an agricultural assistant helping farmers maximize profit based on land, soil, irrigation, and local price trends.
🟩 Step 1: Ask the farmer to enter:
📍 Location (village/town, district, state)
🌾 Land Area (in acres or hectares)
💧 Irrigation Type (e.g., borewell, canal, drip, rainfed)
🧪 Soil Type (e.g., loamy, black, red, sandy, clay)
🟩 Step 2: Based on inputs:
Identify agro-climatically suitable crops.
For each crop:
Estimate growing duration
Estimate yield per acre
Predict modal price at harvest (based on local mandi trends over past 3–5 years)
🟩 Step 3: For top crops, calculate:
Total Expected Yield = yield × land area
Revenue = yield × price
Estimated Costs:
Seed, fertilizer, labor, irrigation
Transport (₹50 per 100 km/quintal for <100 km; adjust for more)
Net Profit = Revenue − Input Cost − Transport Cost
🟩 Step 4: Recommend the most profitable crop: 
Display summary:
Crop Name
Yield
Modal Price
Revenue
Total Cost
Net Profit
Time to Harvest (months)
🟩 Step 5: Suggest 3–4 yearly crop patterns as one-liners:
 (Each pattern = Kharif → Rabi → Zaid)
🌾 Moong → Coriander → Cucumber
🌾 Soybean → Wheat → Bottle Gourd
🌾 Maize → Gram → Watermelon
🌾 Vegetable Mix → Mustard → Fallow
🏆 Select the best pattern based on total net profit and show:
🌿 Best Pattern: Moong → Coriander → Cucumber
🪙 Total Annual Profit Estimate: ₹47,000+
📋 Why: Balanced duration, soil health (pulses), high market value, water use optimization.

Do not show in table format in table give data in bullet point and should be short and precise""")


  model = "gemini-2.5-flash"
  contents = [
    types.Content(
      role="user",
      parts=[
        types.Part.from_text(text=f"""hi""")
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
          title="Agricultural Profit Maximization Assistant",
          type="messages",
          multimodal=True,
      )
  demo.launch(show_error=True)
