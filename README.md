# PPE Detection Model

This project has been developed in line with the Syook AI internship assignment requirements.

Assignment doc link: [link](https://github.com/siddham-jain/ppe-detection-model/blob/master/Syook%20-%20AI%20Internship%20-%20Assignment.pdf)

Project Report: [link](https://github.com/siddham-jain/ppe-detection-model/blob/master/Project_Report_PPE_Detection_Model.pdf)

## Project Structure
| Directory| Content 
|----------|-----
| `dataset_utils` | This directory contains all the scripts i used to modify my dataset. 
| `output` | This directory contains output image when i ran inference on `test_data`
| `runs` | This directory was created during training of ppe detection model. I exported this from colab after i finished with the training of the model.
| `test_data.zip` | This directory comtains data for testing the final pipeline. You can directly upload this file in the streamlit app. It will run inference on all images and display the output images.
| `training` | This directory contains colab notebooks which i used for training my models
| `weights` | This directory contains all the final model which are being used in my `inference.py`
| `dataset.zip` | This is the zip file for datasets which i used for training person detection and ppe detection model


## Setup Instructions
1. Clone this repository
    ```bash
    git clone https://github.com/siddham-jain/ppe-detection-model
    cd ppe-detection-model
    ```
2. Setup virtual environment
    - Using venv
    ```bash
    python3 -m venv <your_env_name>
    source <your_env_name>/bin/activate
    pip install -r requirements.txt
    ```
    - Using Conda
    ```bash
    conda env create -f environment.yml
    conda activate ppe_detect # default name is ppe_detect
    ```
    If you want to change the name of your venv, you can do it by changing the name in `environment.yml`
## Usage Instructions
For running this pipeline you can simply run:
```bash
python3 inference.py <input_dir> <output_dir> <path_to_person_detection_model> <path_to_ppe_detection_models>
```
Person detection and ppe detection model are stored in in weights directory of this project.

If you want to access this app through UI, use:
```bash
streamlit run app.py
```





