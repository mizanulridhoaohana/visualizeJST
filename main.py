import streamlit as st
from PIL import Image
import numpy as np
from skimage import feature
from keras.models import load_model
import joblib
import time

def lbp_texture(image):
    radius = 2
    n_point = radius * 8
    lbp = feature.local_binary_pattern(image, n_point, radius, 'default')
    max_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=max_bins, density=True)
    return hist

def result(prediction):
    if prediction[0] == 0:
        return "Corrosion"
    else:
        return "No Corrosion"

def lbp_prediction(image_path):
    start_time = time.time()  # Record the start time
    im = Image.open(image_path).convert('L')
    data = np.array(im)
    lbp_features = lbp_texture(data)
    loaded_model = joblib.load('./mlp_85_lbp_model.pkl')
    prediction = loaded_model.predict([lbp_features])
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time
    print(f"LBP Prediction Execution Time: {execution_time:.4f} seconds")
    return result(prediction)

def lbp_prediction(image_path):
    start_time = time.time()  # Record the start time
    im = Image.open(image_path).convert('L')
    data = np.array(im)
    lbp_features = lbp_texture(data)
    loaded_model = joblib.load('./mlp_85_lbp_model.pkl')
    prediction = loaded_model.predict([lbp_features])
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time
    st.write(f"LBP Prediction Execution Time: {execution_time:.4f} seconds")
    return result(prediction)

def cnn_prediction(image_path):
    start_time = time.time()  # Record the start time
    model = load_model('./best_model_cnn.h5')
    new_input_size = (256, 256)
    img = Image.open(image_path).convert('RGB')
    img = img.resize(new_input_size)
    img_array = np.expand_dims(np.array(img), axis=0)
    prediction = model.predict(img_array)
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time
    st.write(f"CNN Prediction Execution Time: {execution_time:.4f} seconds")
    return result(prediction)

def main():
    st.title("Comparison MLP vs CNN")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Create two columns for LBP and CNN results
        col1, col2 = st.columns(2)

        # LBP Prediction
        with col1:
            st.subheader("MLP+LBP Model Result:")
            lbp_result = lbp_prediction(uploaded_file)
            st.write(lbp_result)

        # CNN Prediction
        with col2:
            st.subheader("CNN Model Result:")
            cnn_result = cnn_prediction(uploaded_file)
            st.write(cnn_result)

if __name__ == "__main__":
    main()