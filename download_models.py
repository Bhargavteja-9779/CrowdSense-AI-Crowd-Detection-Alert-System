import urllib.request
import os

def download_file(url, filename):
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename}")

# Download YOLO model files
download_file("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg", "yolov3.cfg")
download_file("https://pjreddie.com/media/files/yolov3.weights", "yolov3.weights")
download_file("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names", "coco.names")

print("All files downloaded successfully!") 