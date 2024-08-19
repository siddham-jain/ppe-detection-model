# PPE Detection Model

This project has been developed in line with the Syook AI internship assignment requirements.

Assignment doc link: [link](!https://github.com/siddham-jain/ppe-detection-model/blob/master/Syook%20-%20AI%20Internship%20-%20Assignment.pdf)

Project Report: [link](!https://github.com/siddham-jain/ppe-detection-model/blob/master/Project_Report_PPE_Detection_Model.pdf)

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





