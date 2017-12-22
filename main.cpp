// A minimal but useful C++ example showing how to load an Imagenet-style object
// recognition TensorFlow model, prepare input images for it, run them through
// the graph, and interpret the results.
//
// It's designed to have as few dependencies and be as clear as possible, so
// it's more verbose than it could be in production code. In particular, using
// auto for the types of a lot of the returned values from TensorFlow calls can
// remove a lot of boilerplate, but I find the explicit types useful in sample
// code to make it simple to look up the classes involved.
//
// To use it, compile and then run in a working directory with the
// learning/brain/tutorials/label_image/data/ folder below it, and you should
// see the top five labels for the example Lena image output. You can then
// customize it to use your own models or images by changing the file names at
// the top of the main() function.
//
// The googlenet_graph.pb file included by default is created from Inception.
//
// Note that, for GIF inputs, to reuse existing code, only single-frame ones
// are supported.

#include <fstream>
#include <utility>
#include <vector>
#include <iostream>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.hpp>

#include "utils.h"

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {

    // These are the command-line flags the program can understand.
    // They define where the GRAPH and input data is located, and what kind of
    // input the model expects. If you train your own model, or use something
    // other than inception_v3, then you'll need to update these.
    string imagePath(argv[1]);
    string GRAPH = "models/ssd_mobilenet_v1_coco_+_egohands_+_extended/inference/frozen_inference_graph.pb";
    string LABELS = "data/labels_map.pbtxt";

    int32 inputWidth = 299;
    int32 inputHeight = 299;
    float inputMean = 0;
    float inputStd = 255;

    string inputLayer = "image_tensor:0";
    vector<string> outputLayer = {"detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0"};

    string ROOTDIR = "/data/Y.Disk/work/visme/detector-v-2/";

    // First we load and initialize the model.
    std::unique_ptr<tensorflow::Session> session;
    string graphPath = tensorflow::io::JoinPath(ROOTDIR, GRAPH);
    LOG(INFO) << "graphPath:" << graphPath;
    Status loadGraphStatus = loadGraph(graphPath, &session);
    if (!loadGraphStatus.ok()) {
        LOG(ERROR) << "loadGraph(): ERROR" << loadGraphStatus;
        return -1;
    } else
        LOG(INFO) << "loadGraph(): frozen graph loaded" << endl;


    // Get the image from disk as a float array of numbers, resized and normalized
    // to the specifications the main GRAPH expects.
    std::vector<Tensor> resizedTensors;
    Status readTensorStatus =
            readTensorFromImageFile(imagePath, inputHeight, inputWidth, inputMean,
                                    inputStd, &resizedTensors);
    if (!readTensorStatus.ok()) {
        LOG(ERROR) << readTensorStatus;
        return -1;
    }
    const Tensor& resizedTensor = resizedTensors[0];

    LOG(ERROR) << "image shape:" << resizedTensor.shape().DebugString() << ",len:" << resizedTensors.size() << ",tensor type:" << resizedTensor.dtype();
    // << ",data:" << resizedTensor.flat<tensorflow::uint8>();

    // Actually run the image through the model.
    std::vector<Tensor> outputs;
    Status runStatus = session->Run({{inputLayer, resizedTensor}},
                                     outputLayer, {}, &outputs);
    if (!runStatus.ok()) {
        LOG(ERROR) << "Running model failed: " << runStatus;
        return -1;
    }

    int imageWidth = resizedTensor.dims();
    int imageHeight = 0;
    //int imageHeight = resizedTensor.shape()[1];

    LOG(ERROR) << "size:" << outputs.size() << ",imageWidth:" << imageWidth << ",imageHeight:" << imageHeight << endl;

    //tensorflow::TTypes<float>::Flat iNum = outputs[0].flat<float>();
    tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
    tensorflow::TTypes<float>::Flat numDetections = outputs[3].flat<float>();
    auto boxes = outputs[0].flat_outer_dims<float,3>();

    LOG(ERROR) << "numDetections:" << numDetections(0) << "," << outputs[0].shape().DebugString();

    for(size_t i = 0; i < numDetections(0) && i < 20;++i)
    {
        if(scores(i) > 0.5)
        {
            LOG(ERROR) << i << ",score:" << scores(i) << ",class:" << classes(i)<< ",box:" << "," << boxes(0,i,0) << "," << boxes(0,i,1) << "," << boxes(0,i,2)<< "," << boxes(0,i,3);
        }
    }

    std::map<int, std::string> labelsMap = std::map<int,std::string>();
    for (int i = 0; i < 21; i++)
        labelsMap[i] = "test label #" + to_string(i);

    VideoCapture cap("/data/Y.Disk/work/visme/data/raw/test_left.wmv");
    Mat frame;
    Tensor tensor;
    while (cap.isOpened()) {
        cap >> frame;
        cvtColor(frame, frame, COLOR_BGR2RGB);

        readTensorStatus = readTensorFromMat(frame, 3, tensor);
        if (!readTensorStatus.ok()) {
            LOG(ERROR) << "Mat->Tensor conversion failed: " << readTensorStatus;
            return -1;
        }

        LOG(INFO) << "image shape:" << tensor.shape().DebugString() << ", tensor type:" << tensor.dtype();

        outputs.clear();
        runStatus = session->Run({{inputLayer, tensor}}, outputLayer, {}, &outputs);
        if (!runStatus.ok()) {
            LOG(ERROR) << "Running model failed: " << runStatus;
            return -1;
        }

        //iNum = outputs[0].flat<float>();
        scores = outputs[1].flat<float>();
        classes = outputs[2].flat<float>();
        numDetections = outputs[3].flat<float>();
        boxes = outputs[0].flat_outer_dims<float,3>();

        LOG(INFO) << "numDetections:" << numDetections(0) << "," << outputs[0].shape().DebugString();

        for(size_t i = 0; i < numDetections(0) && i < 20;++i)
            if(scores(i) > 0.5)
                LOG(INFO) << i << ",score:" << scores(i) << ",class:" << classes(i)<< ",box:" << "," << boxes(0,i,0) << "," << boxes(0,i,1) << "," << boxes(0,i,2)<< "," << boxes(0,i,3);

        drawBoundingBoxesOnImage(frame, scores, classes, boxes, labelsMap, 0.5);
        imshow("stream", frame);
        waitKey(10);
    }
    destroyAllWindows();

    return 0;
}