import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Import functions from the modified scripts
from countInPhoto import count_cars_in_photo
from countInVideo import count_vehicles_in_video
from Number_plate import detect_number_plate

# Title
st.title("Vehicle Analysis Web App")
st.write("Choose a functionality below:")

# Sidebar for user selection
option = st.sidebar.radio(
    "Select Functionality",
    ("Count Cars in Photo", "Count Vehicles in Video", "Detect Number Plate")
)

# Upload file
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "png", "mp4"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    if option == "Count Cars in Photo":
        st.subheader("Counting Cars in Image")
        output_image, car_count = count_cars_in_photo(temp_path)
        st.image(output_image, caption="Processed Image", use_column_width=True)
        st.write(f"Number of cars detected: {car_count}")

    elif option == "Count Vehicles in Video":
        st.subheader("Counting Vehicles in Video")
        car_count = count_vehicles_in_video(temp_path)
        st.write(f"Total vehicles detected: {car_count}")

    elif option == "Detect Number Plate":
        st.subheader("Number Plate Detection")
        output_image, plate_text = detect_number_plate(temp_path)
        if output_image is not None:
            st.image(output_image, caption="Detected Number Plate", use_column_width=True)
            st.write(f"Extracted Number Plate: {plate_text}")
        else:
            st.write("No number plate detected.")

    os.remove(temp_path)  # Cleanup temporary file
