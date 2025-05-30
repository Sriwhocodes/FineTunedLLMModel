# 🚗 Indian Used Car Price Predictor

A machine learning web application that predicts the price of a used car in India using a fine-tuned DistilBERT language model. Built with 🤗 Transformers and Gradio.

## 🔍 Live Demo

👉 Try it on Hugging Face Spaces: https://huggingface.co/spaces/coolbro99/car-price-predictor

## 💡 How It Works

This app uses a fine-tuned DistilBERT transformer model (`DistilBertForSequenceClassification`) to predict car prices as a regression task. The model takes a natural language prompt containing structured car information and outputs the predicted price in lakhs of INR.

### 🔍 Example Prompt

Predict price: 2018 Hyundai i20, petrol fuel, manual transmission, first owner, 45,000 km in Delhi.

The model interprets this prompt and returns a price like:  
₹450,000

## 📦 Features

- Fine-tuned DistilBERT model for car price prediction.
- Clean, responsive Gradio interface.
- Dropdown menus for brand, model, fuel, transmission, and more.
- Live examples for quick testing.
- Custom input support for car model names.

## 🧠 Model Details

- Model: DistilBertForSequenceClassification
- Tokenizer: DistilBertTokenizer
- Framework: Hugging Face Transformers
- The model was trained on structured car data converted to natural language format.

Note: The model outputs a single logit, interpreted as the price in lakhs of INR.

## 🛠️ Installation (Local)

1. Clone the repository:
   git clone https://github.com/Sriwhocodes/car-price-predictor.git
   cd car-price-predictor

2. Set up a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Run the app:
   python app.py

## 🖼️ UI Framework

Built with Gradio to quickly deploy and interact with ML models.

## 📁 Project Structure
car-price-predictor/
├── app.py                 # Main application script
├── car_price_model/       # Directory containing the fine-tuned DistilBERT model files
│   ├── pytorch_model.bin  # Model weights
│   ├── config.json        # Model configuration
│   └── tokenizer          # Tokenizer files (e.g., vocab.txt, tokenizer_config.json)
├── requirements.txt       # List of Python dependencies
└── README.md              # Project documentation

## ✍️ Author
- coolbro99 – https://huggingface.co/coolbro99

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for more details.
