# 🌾 FarmBuddy - AI-Powered Personal Assistant for Indian Farmers

**FarmBuddy** is an AI-driven intelligent assistant designed to empower Indian farmers by maximizing their profit, minimizing crop loss, and providing timely, personalized advice. It combines the power of Google Cloud, real-time mandi data, weather forecasting, crop diagnosis, and government scheme recommendations — all in a voice-first experience tailored for both literate and non-literate users.

---

## 🚀 Features

- **Crop Disease Diagnosis**  
  Upload a photo of a diseased crop and get instant diagnosis, local remedies, and severity insights to prevent loss.

- **Market Price Trend Analysis**  
  Analyze historical mandi price data with **Buy / Sell / Hold** recommendations to optimize selling decisions.

- **Government Scheme Discovery**  
  Get personalized suggestions for schemes you’re eligible for — no paperwork or confusion.

- **Best Mandi Recommendation**  
  Suggests mandis offering the highest net profit after calculating transport cost and current modal price from nearby locations.

- **Rain Prediction & Weather Alerts**  
  Weekly rainfall forecasts and advisories to help farmers plan irrigation, sowing, or harvesting schedules.

- **Profit Maximizer (Crop Planner)**  
  Personalized crop planning and rotation strategies based on soil type, location, irrigation, and season to maximize annual earnings.

---

## 🧠 Technologies Used

| Technology        | Purpose                                                                 |
|------------------|-------------------------------------------------------------------------|
| **Vertex AI (Google)**     | Used for grounding responses via search and powering intelligent agent behavior.             |
| **Firebase**      | Handles secure email-password based authentication for farmers.         |
| **Cloud Functions** | Cost-effective microservices backend; Docker-packaged functions invoked only on demand. |
| **Google Maps API** | Calculates distance between mandis and user location for transport cost estimation. |
| **Text-to-Speech / Speech-to-Text** | Enables multilingual, voice-based interaction in Hindi or English.                  |
| **Vertex AI Search** | Powers document-grounded policy recommendations and price analysis.  |

---
![WhatsApp Image 2025-07-27 at 11 10 59_0ed1c178](https://github.com/user-attachments/assets/4c1a2fd6-9f1c-4ca9-a7ff-243b3932d1d5)


---

## ✅ Impact

- Enables data-driven decisions on what and where to sell.
- Minimizes crop loss by early disease detection.
- Reduces dependency on local middlemen and unreliable information.
- Provides transparent access to government schemes and benefits.
- Makes farming more profitable, resilient, and tech-enabled.

---

## 🔍 How Is This Solution Different?

- **Hyperlocal Profit Maximization**: Recommends not just the price but the net income after deducting transport — a game-changer for rural logistics.
- **Personalized Agri-Advisory**: Crop plans, price trends, weather, and subsidies are customized for each farmer.
- **Voice-First & Vernacular**: Designed for low-literate users with speech-based input/output in Hindi or English.
- **Multi-Modal Inputs**: Farmers can send text, images, or voice — and get actionable output across all modes.

---

## 🔮 Future Scope

- AI-driven yield forecasting using satellite and soil data.
- Voice-based WhatsApp alerts for illiterate users.
- Pest outbreak alerts powered by community data and image recognition.
- Integration with eNAM and Kisan Credit Card.
- Regional language expansion beyond Hindi.
- Farmer community insights layer — "what others like you are growing/selling."

---

## 🛠️ Getting Started (Dev Mode)

### Prerequisites

- Python 3.10+
- Docker
- Firebase CLI
- Google Cloud SDK

### Clone the Repository
