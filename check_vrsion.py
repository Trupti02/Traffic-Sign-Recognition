# check_versions.py
import torch
import torchvision
import flask
import numpy
import pandas
import cv2
import sklearn
import PIL

print(f"torch        == {torch.__version__}")
print(f"torchvision  == {torchvision.__version__}")
print(f"flask        == {flask.__version__}")
print(f"numpy        == {numpy.__version__}")
print(f"pandas       == {pandas.__version__}")
print(f"opencv       == {cv2.__version__}")
print(f"scikit-learn == {sklearn.__version__}")
print(f"pillow       == {PIL.__version__}")