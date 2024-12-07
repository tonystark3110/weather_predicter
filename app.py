import streamlit as st
import torch
import numpy as np
from torch import nn

# Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        output = self.fc(h_n[-1])
        return output

# Load the trained model
input_dim = 3
hidden_dim = 64
output_dim = 1
num_layers = 2

model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)
model.load_state_dict(torch.load("best_model_3_features.pth"))
model.eval()

# Streamlit App
st.title("Temperature Prediction App")
st.write("Predict the next hour's temperature based on 24 hours of weather data.")

# User Input
temperature_input = st.text_area("Enter 24 temperature values (comma-separated):")
humidity_input = st.text_area("Enter 24 humidity values (comma-separated):")
wind_speed_input = st.text_area("Enter 24 wind speed values (comma-separated):")

if st.button("Predict"):
    try:
        # Parse inputs
        temp_list = [float(x.strip()) for x in temperature_input.split(",")]
        humidity_list = [float(x.strip()) for x in humidity_input.split(",")]
        wind_speed_list = [float(x.strip()) for x in wind_speed_input.split(",")]

        if len(temp_list) != 24 or len(humidity_list) != 24 or len(wind_speed_list) != 24:
            st.error("Please provide exactly 24 values for each feature.")
        else:
            # Prepare input tensor
            input_data = np.array([temp_list, humidity_list, wind_speed_list]).T.reshape(1, -1, 3)
            input_tensor = torch.tensor(input_data, dtype=torch.float32)

            # Make prediction
            prediction = model(input_tensor).item()
            st.success(f"Predicted Temperature: {round(prediction, 2)}°C")
    except Exception as e:
        st.error(f"Error: {e}")
