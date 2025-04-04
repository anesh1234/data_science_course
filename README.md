# CS4020 Project Spring 25
## Setting up the environment
A requirements.txt file was made so that anyone easily can reproduce our Python environment. We have used Python 3.12 for this project, and working with environments in this version is detailed at [Python Docs](https://docs.python.org/3.12/tutorial/venv.html).
In short, to create a compatible Python 3.12 virtual environment for this project, do the following:
Create a new virtual environment
```Shell
python3.12 -m venv .venv
```

Activate the environment. If you are working in VSCode, please make sure you are using Command Prompt and not PowerShell as your terminal, as PowerShell may prevent you from activating the environment.
```Shell
.venv\Scripts\activate
```

Import the requirements into the environment:
```Shell
pip install -r requirements.txt
```

To enable training on GPU, you will need a CUDA enabled GPU, wchich can be checked at [NVIDIA's page](https://developer.nvidia.com/cuda-gpus), with its latest driver.
Further, the pytorch library must be installed in a [specific way](https://pytorch.org/get-started/locally), this is already in the requirements.txt, but might change over time.

## File Structure
The [YOLO](https://github.com/anesh1234/data_science_course/tree/main/YOLO) folder contains files related to the YOLO models directly, including the scripts used for training/validation/tuning/inference, and folder structures for storing results from running these scripts. Each model configuration we trained has its own folder, which contains their respective training results, and the best model achieved within the /weights subfolders. These subfolders are named with IDs, which correspond to the training iterations mentioned in our report. This is also true for the tuning runs. The .yaml files contain the dynamic training parameters for each configuration.

The /datasets folder contains the individual datasets we used: /d1, /d2 and /d3. Further, it contains the /final folder, which keeps the merged YOLO-formated dataset we trained our models on.

The /tools folder contains scripts which were used to manipulate or gain insights about the datasets.

Note: only relevant files are shown.

```
├── README.md
├── class_ocurrences.xlsx       Class ocurrences per dataset
├── requirements.txt            
├── DETR/                       Legacy folder
│   └── train.py
├── YOLO/                       Contains the programs used to run the model
│   ├── predict.py
│   ├── train.py
│   ├── tune.py
│   ├── validate.py
│   └ ...                       The rest is not shown 
├── datasets/
│   ├── coco/d1/
│   ├── d1/
│   │   ├── data.yaml
│   │   ├── train/
│   │   ├── test/
│   │   └── valid/
│   ├── d2/
│   │   ├── data.yaml
│   │   ├── train/
│   │   ├── test/
│   │   └── valid/
│   ├── d3/
│   │   ├── data.yaml
│   │   ├── train/
│   │   ├── test/
│   │   └── valid/
│   ├── final/
│   │   ├── data.yaml
│   │   ├── train/
│   │   ├── test/
│   │   └── valid/
│   └── processing/             Folder used when merging the datasets
│       └── ...
└── tools/                      Various programs used to process the datasest or extract information
    ├── convert_d2_ann.py
    ├── count_class_occurence.py
    ├── get_max_dims.py
    ├── merge_datasets.py
    ├── old_config.py
    ├── old_resize_radiography.py
    └── old_show_radiography.py
```

## Inference
If you want to test our most successful model, this can be done with the [YOLO/predict.py](https://github.com/anesh1234/data_science_course/blob/main/YOLO/predict.py) script.
