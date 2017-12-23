# tensorflow-object-detection-cpp
A C++ example of running TensorFlow Object Detection model in live mode.
Inspired by [issue](https://github.com/tensorflow/models/issues/1741#issuecomment-318613222).

Requirements (for use without Bazel):
* `TensorFlow` .so builds ([more](https://github.com/tensorflow/tensorflow/issues/2412#issuecomment-300628873), requires Bazel to build)
* `Eigen3` headers ([more](http://eigen.tuxfamily.org/index.php?title=Main_Page))
* `OpenCV` ([more](https://github.com/opencv/opencv))

Usage:
1. Specify your own paths for necessary libs in `CmakeLists.txt`
2. Specify your own paths for `frozen_inference_graph.pb` and `labels_map.pbtxt` in `main.cpp` (lines 44-47)
3. Specify your video source (`main.cpp`, line 80)
4. Have fun

`demo/` dir contains frozen graph & labels map from [victordibia/handstracking](https://github.com/victordibia/handtracking) as an example. 
Feel free to add your own models and contribute.
