# Deep Learning School Detection Project
This repo contains code for final project of Deep Learning School.

# Example
![](https://github.com/vkurichenko/dls-detection-project/blob/main/example.gif)

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
1. Web page will be changed so that its element fit different screen sizes.
2. Docker image will be created, so that the app runs in isolated container.
