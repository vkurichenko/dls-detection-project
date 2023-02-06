# Deep Learning School Detection Project
This repo contains code for final project of Deep Learning School.

## Description
The aim of the project was to create a web app for Neural Detection. The app uses two models, [SSDlite](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html#torchvision.models.detection.ssdlite320_mobilenet_v3_large) and [YOLOv5](https://pytorch.org/hub/ultralytics_yolov5/), that were trained on ~80 different classes. The user can choose the score threshold for each model, which is the confidence of the model for a class to be included in prediction.

## Example
![](https://github.com/vkurichenko/dls-detection-project/blob/main/example.gif)

## Project structure
- `app` folder contains files for the Flask app.
    - `main.py` is the app itself.
    - `templates/index.html` is the web representation of the app.
- `notebooks` contains Jupyter Notebooks.
    - `object_detection_fall_2021.ipynb` is the course creators' notebook with the project requirements.
    - `DLS_project.ipynb` is my notebook with the project description based on the requirements.
- `example.gif` is an example of app running on `test-image.jpg`.
- `requirments.txt` contains dependencies.

## Running the app in a Docker container
1. Run `docker run --rm --name dls-detection-app -p 5050:5050 vkurichenko/dls-detection:latest` in order to run new container named `dls-detection-app` from `vkurichenko/dls-detection:latest` image located on Docker Hub, exposing `5050` port for the app.
2. Open `http://0.0.0.0:5050`.
3. Upload your image, set the models threshold, press `Predict Image`.
4. Images are saved to `/app/static/images`. Also you can download them via `Download` button.
5. Enjoy!

## Manual installation
1. Create new virtual environment via `conda` or `pyenv` (tested on `Python 3.9.15`).
2. Install all dependencies from `requirments.txt` file (exact versions while testing are listed there).
3. Go to `/app` folder.
4. Run `python main.py`.
5. Open `http://0.0.0.0:5050`.
6. Upload your image, set the models threshold, press `Predict Image`.
7. Images are saved to `/app/static/images`. Also you can download them via `Download` button.
8. Enjoy!

## Future improvements
1. Web page will be changed so that its elements fit different screen sizes.
2. App code will be changed so that the first run of the image detection does not take longer than consequent ones.
