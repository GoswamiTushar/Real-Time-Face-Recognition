# **Name**
Real Time Face Dectector
Python model to detect the faces of people provided in the dataset.

# **Description**
This model has following key steps:
	- This project generates the dataset of our own by recording the the video and getting their label.
  - On the basis of the generated dataset the model searches for the people's from the real time video.
  - Label the people in the video according to the trained dataset with their names hovering with the rectangle floating around their face.

# **Prerequisites**
`Pyhton3` installed with following libraries on your system.

# **Libraries needed**
1. `cv2` for computer vision on images
    - `Install` with following command `pip install opencv-python` and `pip install numpy`
2. `numpy` for vedio and image processing
    - `Install` with following command `pip install numpy`
3. `os` for taking camera input from the machine
    - `Install` with following command `pip install os-sys`
    
# **How to run** 
After the installation of all packages run with the following command
1. `Create` a folder under the root of the project named `images` which will use store the dataset on which the model will be trained.
2. Generate your own dataset by executing the file `Generate_Dataset.py` by the following command `python Generate_Dataset.py` in the terminal.
3. To quit from the the process of dataset generation click on the dialog box showing the camera input and press `q`.
4. After successful dataset generation, execute the file `face_recognition.py` by the following command `python face_recognition.py` in the terminal.
5. Dialog box opens showing the camera input with the rectangular boxes showing around the people whose dataset is present.
6. To quit from the the process of execution click on the dialog box showing the camera input and press `q`.

# **Disclaimer**
This project was by [`ardourApeX`](https://github.com/ardourApeX) and [`CyberWake`](https://github.com/CyberWake) to detect faces in dataset and mark them respectively with there names.