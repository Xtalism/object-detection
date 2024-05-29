This a Yolo V5 object detection project. I used Python 3.8 to write this program.

In order to install the necessary dependencies we use requirements.txt file and build our Python 3.8 virtual environment. 
Just as follows:

    py -3.8 -m venv {namefyourvirtualenvironment}
    py -m pip install requirements.txt

If you happen to change or add new dependencies, you can also make your own requirements.txt file as follows:

    py -m pip freeze --local > requirements.txt

You can make your own AI model with Yolov8 official website which includes a detail tutorial on how to do it [yolov5](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/) and you can also check its proper documentation [here]https://github.com/ultralytics/ultralytics