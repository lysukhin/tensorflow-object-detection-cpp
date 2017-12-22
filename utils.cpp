#include "utils.h"

#include <math.h>
#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <regex>

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

#include <cv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

/** Read a model graph definition (xxx.pb) from disk, and creates a session object you can use to run it.
 */
Status loadGraph(const string &graph_file_name,
                 unique_ptr<tensorflow::Session> *session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
            ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}

/** Read a labels map file (xxx.pbtxt) from disk to translate class numbers into human-readable labels.
 */
Status readLabelsMapFile(const string &fileName, map<int, string> &labelsMap) {

    // Read file into a string
    ifstream t(fileName);
    if (t.bad())
        return tensorflow::errors::NotFound("Failed to load labels map at '", fileName, "'");
    stringstream buffer;
    buffer << t.rdbuf();
    string fileString = buffer.str();

    // Split it by ',' and parse each entry
    size_t pos = 0;
    string token;
    smatch matcher;
    const regex reId("[0-9]+");
    const regex reName("\'[A-Z-]+\'");

    int id;
    string name;

    while ((pos = fileString.find(",")) != string::npos) {
        token = fileString.substr(0, pos);
        if (regex_search(token, matcher, reId)) {
            id = stoi(matcher[0].str());
        } else
            continue;
        if (regex_search(token, matcher, reName)) {
            name = matcher[0].str().substr(1, matcher[0].str().length() - 2);
        } else
            continue;
        fileString.erase(0, pos + 1);
        labelsMap.insert(pair<int, string>(id, name));
    }
    return Status::OK();
}

/** Convert Mat image into tensor of shape (1, height, width, d) where last three dims are equal to the original dims.
 */
Status readTensorFromMat(Mat mat, int inputDepth, Tensor &tensor) {

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;

    tensorflow::TensorShape shape = tensorflow::TensorShape();
    shape.AddDim(1);
    shape.AddDim(mat.rows);
    shape.AddDim(mat.cols);
    shape.AddDim(3);
    tensor = Tensor(tensorflow::DT_FLOAT, shape);

    Mat mat32f;
    mat.convertTo(mat32f, CV_32FC1);
    auto matData = (float*) mat32f.data;
    auto inputTensorMapped = tensor.tensor<float, 4>();
    for (int y = 0; y < mat32f.rows; ++y) {
        auto source_row = matData + (y * mat32f.cols * inputDepth);
        for (int x = 0; x < mat32f.cols; ++x) {
            const float* source_pixel = source_row + (x * inputDepth);
            for (int c = 0; c < inputDepth; ++c) {
                const float* source_value = source_pixel + c;
                inputTensorMapped(0, y, x, c) = *source_value;
            }
        }
    }

    auto input_tensor = Placeholder(root.WithOpName("input"), tensorflow::DT_FLOAT);
    vector<pair<string, tensorflow::Tensor>> inputs = {{"input", tensor}};
    auto uint8_caster = Cast(root.WithOpName("uint8_cast"), tensor, tensorflow::DT_UINT8);

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    vector<Tensor> out_tensors;
    unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));

    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {"uint8_cast"}, {}, &out_tensors));

    tensor = out_tensors.at(0);
    return Status::OK();
}

/** Draw bounding box and add caption to the image.
 *  Boolean flag _scaled_ shows if the passed coordinates are in relative units (true by default in tensorflow detection)
 */
void drawBoundingBoxOnImage(Mat &image, double yMin, double xMin, double yMax, double xMax, double score, string label, bool scaled=true) {
    cv::Point tl, br;
    if (scaled) {
        tl = cv::Point((int) (xMin * image.cols), (int) (yMin * image.rows));
        br = cv::Point((int) (xMax * image.cols), (int) (yMax * image.rows));
    } else {
        tl = cv::Point((int) xMin, (int) yMin);
        br = cv::Point((int) xMax, (int) yMax);
    }
    cv::rectangle(image, tl, br, cv::Scalar(0, 255, 255), 1);

    // Ceiling the score down to 3 decimals (weird!)
    float scoreRounded = floorf(score * 1000) / 1000;
    string scoreString = to_string(scoreRounded).substr(0, 5);
    string caption = label + " (" + scoreString + ")";

    // Adding caption of type "LABEL (X.XXX)" to the top-left corner of the bounding box
    int fontCoeff = 12;
    cv::Point brRect = cv::Point(tl.x + caption.length() * fontCoeff / 1.6, tl.y + fontCoeff);
    cv::rectangle(image, tl, brRect, cv::Scalar(0, 255, 255), -1);
    cv::Point textCorner = cv::Point(tl.x, tl.y + fontCoeff * 0.9);
    cv::putText(image, caption, textCorner, FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0));
}

/** Draw bounding boxes and add captions to the image.
 *  Box is drawn only if corresponding score is higher than the _threshold_.
 */
void drawBoundingBoxesOnImage(Mat &image,
                              tensorflow::TTypes<float>::Flat scores,
                              tensorflow::TTypes<float>::Flat classes,
                              tensorflow::TTypes<float,3>::Tensor boxes,
                              map<int, string> labelsMap, double threshold=0.5) {
    for (int j = 0; j < scores.size(); j++)
        if (scores(j) > threshold)
            drawBoundingBoxOnImage(image, boxes(0,j,0), boxes(0,j,1), boxes(0,j,2), boxes(0,j,3), scores(j), labelsMap[classes(j)]);
}