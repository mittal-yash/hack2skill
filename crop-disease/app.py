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
  msg1_image1 = types.Part.from_bytes(
      data=base64.b64decode("""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"""),
      mime_type="image/jpeg",
  )
  msg2_text1 = types.Part.from_text(text=f"""This looks like **Blight** on your tomatoes. 

**Severity:** The image suggests a **Medium** level of infection. You can see significant white, fuzzy growth on some of the tomatoes, indicating the disease is present and spreading.

**Profit Loss:** Blight can cause significant profit loss. It can lead to yield reduction, as the disease weakens the plant and affects fruit quality. In severe cases, it can lead to a **30-50% yield reduction** or even total crop loss if not managed.

**Low-Cost, Local Treatments:**

Here are some simple and affordable treatments you can try:

1.  **Remove Affected Parts:** Immediately remove any tomatoes or leaves showing signs of blight. Dispose of them away from your garden to prevent further spread.
2.  **Baking Soda Spray:** Mix 1.5 tablespoons of baking soda with 1.5 tablespoons of vegetable oil and 1 teaspoon of dish soap in 1 gallon of water. Spray this mixture on the affected plants. Reapply weekly or after rain.
3.  **Hydrogen Peroxide Spray:** Mix 8-12 tablespoons of hydrogen peroxide with 1 gallon of water. Start with a lower concentration and test on a small area first, as too much can harm the plant. Spray all parts of the plant.
4.  **Mulching:** Apply mulch around the base of your tomato plants. This helps prevent soil-borne fungal spores from splashing onto the leaves and fruits.
5.  **Improve Air Circulation:** Ensure your plants have enough space between them to allow for good airflow. This reduces humidity and makes it harder for the blight to spread.
6.  **Water at the Base:** When watering, try to water the soil directly at the base of the plant, rather than overhead. This keeps the leaves dry and less susceptible to fungal infections.

Remember, early detection and quick action are key to managing blight and protecting your tomato harvest!""")
  msg3_image1 = types.Part.from_bytes(
      data=base64.b64decode("""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"""),
      mime_type="image/jpeg",
  )
  si_text1 = types.Part.from_text(text=f"""You are an expert crop doctor trained on images of Indian crops and local farming knowledge.
 Given a photo of a diseased crop, do the following:
Identify the disease affecting the crop.
Rate the severity of the disease: Low, Medium, or High.
Predict if the disease could lead to profit loss and give a rough estimate (e.g., minor loss, 30% yield reduction, etc.).
Recommend low-cost, local treatments or home remedies that are easily available to Indian farmers.
Keep your language simple, actionable, and support Hindi/English as per user input and give short answers""")


  model = "gemini-2.5-flash-lite"
  contents = [
    types.Content(
      role="user",
      parts=[
        msg1_image1,
        types.Part.from_text(text=f"""what happened to this?""")
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
        msg3_image1,
        types.Part.from_text(text=f"""what happened""")
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
                title="Crop Disease Diagnosis",
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
