import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import gradio as gr

# Price formatting function
def format_price(price_in_lakhs):
    rupees = price_in_lakhs * 100000
    return f"₹{rupees:,.0f}"

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DistilBertForSequenceClassification.from_pretrained("car_price_model").to(device)
tokenizer = DistilBertTokenizer.from_pretrained("car_price_model")

# Sample data for dropdowns (replace with your actual data)
top_brands = ["Maruti", "Hyundai", "Honda", "Toyota", "Ford"]
top_models = ["Swift Dzire VXI", "Creta 1.6 SX", "City i-VTEC VX", "Innova Crysta", "EcoSport"]

# PREDICTION FUNCTION WITH INTEGER CONVERSION
def predict_price(year, brand, model_name, km, fuel, transmission, owner, location):
    # Convert inputs to proper types
    year = int(year)
    km = int(km)
    
    prompt = f"Predict price: {year} {brand} {model_name}, {fuel.lower()} fuel, " \
             f"{transmission.lower()} transmission, {owner.lower()} owner, " \
             f"{km:,} km in {location}."

    inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    return format_price(outputs.logits.item())

# Create UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚗 Indian Used Car Price Predictor")
    
    with gr.Row():
        with gr.Column():
            year = gr.Slider(2000, 2023, value=2018, step=1, label="Year of Registration")
            brand = gr.Dropdown(top_brands, label="Brand", value="Hyundai")
            model_name = gr.Dropdown(top_models, label="Model", value="i20")
            km = gr.Slider(1000, 200000, value=45000, label="Kilometers Driven", step=1000)

        with gr.Column():
            fuel = gr.Dropdown(["Petrol", "Diesel", "CNG", "LPG", "Electric"], label="Fuel Type", value="Petrol")
            transmission = gr.Dropdown(["Manual", "Automatic"], label="Transmission", value="Manual")
            owner = gr.Dropdown(["First", "Second", "Third", "Fourth & Above"], label="Owner Type", value="First")
            location = gr.Dropdown(["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
                                  "Kolkata", "Pune", "Ahmedabad"], label="Location", value="Delhi")

    submit_btn = gr.Button("🚀 Predict Price", variant="primary")
    output = gr.Label(label="Predicted Price")

    submit_btn.click(
        fn=predict_price,
        inputs=[year, brand, model_name, km, fuel, transmission, owner, location],
        outputs=output
    )

    gr.Examples(
        examples=[
            [2017, "Maruti", "Swift Dzire VXI", 35000, "Petrol", "Manual", "First", "Mumbai"],
            [2019, "Hyundai", "Creta 1.6 SX", 18000, "Diesel", "Automatic", "First", "Bangalore"],
            [2015, "Honda", "City i-VTEC VX", 42000, "Petrol", "Manual", "Second", "Delhi"]
        ],
        inputs=[year, brand, model_name, km, fuel, transmission, owner, location]
    )

# Launch UI
if __name__ == "__main__":
    demo.launch()