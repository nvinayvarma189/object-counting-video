# object-counting-video

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

A simple program to count the number of instances of one or more objects present in a video


### Built With

* [YoloV3](https://arxiv.org/abs/1804.02767)
* [OpenCV](https://opencv.org/)
* [Python3](https://www.python.org/)

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running:.

### Prerequisites

1. `git clone https://github.com/nvinayvarma189/object-counting-video`
2. `cd object-counting-video && mkdir input/ && mkdir output/` 
3. Downlaod a traffic video clip such as [this](https://www.youtube.com/watch?v=jjlBnrzSGjc) and place it unde `input` folder under your `object-counting-video` folder.
4. Downlaod [Yolov3 weights](https://pjreddie.com/media/files/yolov3.weights) file and place in under [`yolo-coco`](https://github.com/nvinayvarma189/object-counting-video/tree/main/yolo-coco) folder.


### Installation

1. `python -m venv env`
2. `source env/bin/activate`
3. `pip install -r requirements.txt`

### Usage

1. If you have the above mentioned prerequisites and have followed the installation steps then you can run this command 
      - `python main.py count_objects_in_video`
2. If you want to pass any arguments to the script, you can do pass them like this
      - `python main.py count_objects_in_video --input_video_path='/some/custom/path`
      - `python main.py count_objects_in_video --line_coords='[(x1, y1,), (x2, y2)]'`
      - `python main.py count_objects_in_video --output_json_path='/some/custom/path/output.json`
      - `python main.py count_objects_in_video --objects_to_count='["object1", "object2", "object3"]`
3. To list all possible arguments alogn with their default values:
      - `python main.py -- --help`

### Contributing

Coming Soon....

### License

This repository is licensed with the [MIT License](https://github.com/nvinayvarma189/object-counting-video/blob/main/LICENSE)

### References

- [YoloV3 paper](https://arxiv.org/abs/1804.02767) and [Pretrained YoloV3 weights](https://pjreddie.com/media/files/yolov3.weights)
- [Object Detection and Tracking blog post](https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98)
- [SORT: A Simple, Online and Realtime Tracker](https://github.com/abewley/sort/blob/master/sort.py)
