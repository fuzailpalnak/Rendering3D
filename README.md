# Augmented Reality Rendering using Camera

## Description

The goal of this project is to implement a camera pose estimator based on natural feature tracking, with the RANSAC algorithm to enhance the accuracy of the pose estimation and render a 3d-cube. 


## Steps

### 1. Feature Detection and Description

In this step, the application detects and describes features from the reference frame. It also identifies the probable matches of these features on the input frame.

### 2. Feature Matching

To reduce the number of false positives, a feature matching step is performed. Since there may be numerous features with potential false matches, this step aims to refine the matches for better accuracy.

### 3. Pose Estimation

After the feature matching process, the next step is pose estimation. Here, the application assumes that the matched features may contain outliers. To obtain a reliable estimate, the Random Sample Consensus (RANSAC) algorithm is employed. RANSAC helps in finding a robust pose estimate that is not significantly affected by outliers.

### 4. Rendering

Using the estimated pose from the previous step, the application proceeds to render a 3D object. This final step visualizes the object based on the obtained pose information, resulting in a realistic representation.

## Installation

1. Clone the repository: `git clone https://github.com/fuzailpalnak/3dRendering.git`
2. Navigate to the project directory: `cd project_directory`
3. Install the required dependencies: `pip install -r requirements.txt`

## Usage
1. Follow [data creation](https://github.com/fuzailpalnak/3dRendering/files/11832763/dataCreation.pdf) steps to render on custom objects
2. Run
    ```python
    python rendering/ar.py
    ```





https://github.com/fuzailpalnak/3dRendering/assets/24665570/e74329c5-8f33-45e6-a4c4-6dd0cdb596fc


   
