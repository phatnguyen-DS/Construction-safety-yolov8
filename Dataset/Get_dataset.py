from roboflow import Roboflow
import os

# Tải data từ Roboflow
rf = Roboflow(api_key=os.getenv("ROBoflow_API_KEY"))
project = rf.workspace("phat-tlik5").project("phat_project-mlezd")
version = project.version(3)
dataset = version.download("yolov8")