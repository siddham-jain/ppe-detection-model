import streamlit as st
from inference import load_models, process_image
import os
import tempfile

base_dir = os.path.dirname(os.path.abspath(__file__))

project_dir = os.path.join(base_dir, "weights")
person_model_path = os.path.join(project_dir, "person.pt")
ppe_models_dir = os.path.join(project_dir, "ppe_models")


st.title("PPE Detection App")

input_option = st.selectbox("Select Input Type", ["Single Image", "Image Directory"])

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
                process_image(temp_path, person_model, ppe_models, output_dir)
                
                output_image_path = os.path.join(output_dir, os.path.basename(temp_path))
                st.image(output_image_path, caption='Processed Image', use_column_width=True)
        
        st.success("Processing complete!") 

elif input_option == "Image Directory":
    uploaded_dir = st.file_uploader("Upload a directory of images...", type=["zip"], accept_multiple_files=False)
    if uploaded_dir is not None:
        with st.spinner("Processing images..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                upload_temp_path = os.path.join(temp_dir, uploaded_dir.name)
                with open(upload_temp_path, "wb") as f:
                    f.write(uploaded_dir.read())

                output_dir = tempfile.mkdtemp()
                os.system(f'unzip {upload_temp_path} -d {temp_dir}')
                
                person_model, ppe_models = load_models(person_model_path, ppe_models_dir)
                image_files = [f for f in os.listdir(temp_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
                for image_file in image_files:
                    image_path = os.path.join(temp_dir, image_file)
                    process_image(image_path, person_model, ppe_models, output_dir)
                processed_images = [os.path.join(output_dir, img) for img in os.listdir(output_dir)]
                for img in processed_images:
                    st.image(img, caption=os.path.basename(img), use_column_width=True)

        st.success("Processing complete!")