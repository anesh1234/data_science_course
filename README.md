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
The [YOLO](https://github.com/anesh1234/data_science_course/tree/main/YOLO) folder contains files related to the YOLO model directly, including the train.py script, the predict.py script (perform inference with a model) and folder structures for storing results from running the training. Each model configuration we trained has its own folder, which contains their respective training results.