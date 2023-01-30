# Deep Learning School Detection Project
This repo contains code for final project of Deep Learning School.

# Example
![](https://github.com/vkurichenko/dls-detection-project/blob/main/example.gif)

# Project structure
- `app` folder contains files for the Flask app.
    - `main.py` is the app itself.
    - `templates/index.html` is the web representation of the app.
- `notebooks` contains Jupyter Notebooks.
    - `object_detection_fall_2021.ipynb` is the course creators' notebook with the project requirements.
    - `DLS_project.ipynb` is my notebook with the project description based on the requirements.
- `example.gif` is an example of app running on `test-image.jpg`.
- `requirments.txt` contains dependencies.

# Installation
1. Create new virtual environment via `conda` or `pyenv` (tested on `Python 3.9.15`).
2. Install all dependencies from `requirments.txt` file (exact versions while testing are listed there).
3. Go to `/app` folder.
4. Run `python main.py`.
5. Open `http://0.0.0.0:5050`.
6. Upload your image, set the models threshold, press `Predict Image`.
7. Images are saved to `/app/static/images`. Also you can download them via `Download` button.
8. Enjoy!

# Future improvements
1. Web page will be changed so that its elements fit different screen sizes.
2. Docker image will be created, so that the app runs in isolated container.
