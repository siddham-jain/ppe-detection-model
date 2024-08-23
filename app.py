import streamlit as st
from inference import load_models, process_image
import os
import tempfile
import zipfile

base_dir = os.path.dirname(os.path.abspath(__file__))

project_dir = os.path.join(base_dir, "weights")
person_model_path = os.path.join(project_dir, "person.pt")
ppe_models_dir = os.path.join(project_dir, "ppe_models")
class_mapping = {
        0: 'hard-hat',
        1: 'gloves',
        2: 'mask',
        3: 'glasses',
        4: 'boots',
        5: 'vest',
        6: 'ppe-suit'
    }

st.title("PPE Detection App")

input_option = st.selectbox("Select Input Type", ["Single Image", "Image Directory"])

def list_files(directory):
    """ Recursively list all files in the directory. """
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

if input_option == "Single Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with st.spinner("Processing image..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                person_model, ppe_models = load_models(person_model_path, ppe_models_dir)
                output_dir = tempfile.mkdtemp()
                process_image(temp_path, person_model, ppe_models, output_dir, class_mapping)
                
                output_image_path = os.path.join(output_dir, os.path.basename(temp_path))
                if os.path.exists(output_image_path):
                    st.image(output_image_path, caption='Processed Image', use_column_width=True)
                else:
                    st.error("Failed to process the image.")

        st.success("Processing complete!") 

elif input_option == "Image Directory":
    uploaded_dir = st.file_uploader("Upload a directory of images...", type=["zip"], accept_multiple_files=False)
    if uploaded_dir is not None:
        with st.spinner("Processing images..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                upload_temp_path = os.path.join(temp_dir, uploaded_dir.name)
                with open(upload_temp_path, "wb") as f:
                    f.write(uploaded_dir.read())

                with zipfile.ZipFile(upload_temp_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                all_files = list_files(temp_dir)

                image_files = [f for f in all_files if f.lower().endswith(('png', 'jpg', 'jpeg'))]

                if not image_files:
                    st.error("No images found in the uploaded directory.")
                else:
                    person_model, ppe_models = load_models(person_model_path, ppe_models_dir)
                    output_dir = tempfile.mkdtemp()
                    
                    for image_file in image_files:
                        process_image(image_file, person_model, ppe_models, output_dir, class_mapping)
                    
                    processed_images = [os.path.join(output_dir, img) for img in os.listdir(output_dir)]
                    if not processed_images:
                        st.error("No processed images found.")
                    else:
                        for img in processed_images:
                            st.image(img, caption=os.path.basename(img), use_column_width=True)

        st.success("Processing complete!")
