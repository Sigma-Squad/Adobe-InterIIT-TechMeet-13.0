
# Problem Statement

The Problem Statement, given by Adobe as part of Inter IIT Tech Meet 13.0 addresses challenges in the domains of Image Classification, Artefact Detection and Explainable AI. We aim to accomplish the following tasks:
- Detect AI-generated images.
- Identify artefacts in AI-generated images and generate appropriate explanations.







## Getting Started

- The required python scripts for the solution will be executed using the Visual Studio (VS) Code IDE (recommended) or in the terminal window itself. Running these scripts requires the following steps to be completed:

    - Installation of a python interpreter, which supports a Python version of 3.10 or earlier.

    - Installation of the VS Code Python extension.

    - Creation of a workspace folder, via the execution of the *mkdir* command in the terminal.
    ```bash 
        mkdir adobe_team_97
    ```

- The user changes directory to the folder by running the *cd* command, followed by the creation of a virtual environment (venv) named **venv**.
    ```bash
        cd adobe_team_97
        python3 -m venv venv
    ```

- Enable **Long Path Support** on Windows:
    - Press *Win + R*, type *regedit*, and press *Enter* to open the **Registry Editor**.
    - Navigate to the following path: 
    ```bash
        HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
    ```
    - Find the key named *LongPathsEnabled*.
    - Double-click it and set its value to **1**.
    - Click *OK* and close the **Registry Editor**.

- Activate the virtual environment by running the appropriate command based on your operating system.
    - **Windows**:

        ``` bash
        venv\Scripts\activate.ps1
        ```
    - **macOS/Linux/WSL/Git Bash**:

        ```bash
        source venv/bin/activate
        ```

- Select the Virtual Environment:
    - In VS Code, click on the Python interpreter displayed in the bottom-left corner of the window (or use *Ctrl + Shift + P* and select *"Python: Select Interpreter"*).
    - Choose the interpreter from the virtual environment folder (e.g., *venv*).


## Installation

- The submitted folder needs to be installed and a copy of the folder must be saved in the **adobe_team_97** directory.

- Change user directory to the downloaded sub-folder using the *cd* command.
```bash
    cd ./<sub_folder>
```

- Ensure that pip is upgraded to the latest version
```bash
    python.exe -m pip install --upgrade pip
```

- The required dependenices can be installed by running the **requirements.txt** file using the *pip* command.

```bash
    python3 -m pip install --no-cache-dir -r .\requirements.txt
```

- Create a directory named **gradcam_images** using the *mkdir* command

```bash
    mkdir gradcam_images
```
    
## Execution

-  Run the **Inference Notebook** from the VS Code terminal by executing the *python3* command.

```bash
    python3 adobe_team97_inference_notebook.py
```
- Run the **app.py** file via the *python3* command to run flask.

```bash
    python3 app.py
```

## Outputs and Results

- The output printed in the terminal constitutes the results printed in the following format.

```javascript
    Grad-CAM visualization saved at grad_cam_output_32x32.jpg
    Processed <image_path>:{image_class}
```

- The parameter *image_class* is either *real* or *fake* based on the authenticity of the given image.

- The complete results of the predictions for task 1 are saved in a .json file named **task1_predictions.json**.

- The complete results of the predictions for task 2 are saved in a .json file named **task2_explanations.json**.



