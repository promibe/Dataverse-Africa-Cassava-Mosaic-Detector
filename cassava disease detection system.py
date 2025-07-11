import streamlit as st
import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

#Tensorflow model prediction
# Function to predict on a single image

class_names = ['cassava_mosaic', 'healthy_cassava', 'non_cassava']
model = tf.keras.models.load_model('cassava_classifier_final.h5')

def model_prediction(image_path):
    """
    Predict the class of a single image
    """
    IMG_SIZE = (224, 224)
    img = keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
    img_array = img_array / 255.0  # Normalize
    
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return predicted_class, confidence, predictions[0]



#page title
st.set_page_config(page_title="CASSAVA Disease Detector")

#sidebar
st.sidebar.title('Dashboard')
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

#home page
if app_mode == 'Home':
    st.header("CASSAVA DISEASE RECOGNITION SYSTEM")
    image_path = 'image.png'
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Cassava Disease Recognition System! 🌿🔍
    
    Our mission in the Dataverse Research Institute is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect if it has the cassava mosaic diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the *Cassava Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    
    
    """)

#About Page
if app_mode == 'About':
    st.header("About")
    st.markdown("""
        #### About Dataset
        This dataset is recreated using offline augmentation from the original dataset. The original dataset used in the Project was provided by Dataverse Africa Research Institute. This dataset consists of about 15K rgb images of healthy and diseased cassava leaves which is categorized into 3 different classes. The total dataset is divided into 70, 20, 10 ratio of training, validation, and testing set preserving the directory structure. A new directory containing 2077 test images is created later for prediction purpose.
        #### Content
        1. Training samples: 9939
        2. Validation samples: 3909
        3. Test samples: 2077
                
        Classes: {'cassava_mosaic': 0, 'healthy_cassava': 1, 'non_cassava': 2}
    """)

#Prediction page
if app_mode == 'Disease Recognition':
    st.header('Disease Recognition')
    text_image = st.file_uploader("Choose an Image")
    if st.button("Show Image"):
        if text_image is not None:
            st.image(text_image, use_column_width=True)
        else:
            st.warning("⚠️ Kindly upload an image before clicking the button.")

    #predict button
    if st.button('Predict'):
        with st.spinner('Please wait...'):
            if text_image is not None:
                result_index = model_prediction(text_image)
                st.image(text_image, use_column_width=True)
                st.success(f"Our Model is Predicting its a {class_names[result_index]}")
            else:
                st.warning("⚠️ Kindly upload an image before clicking the button.")