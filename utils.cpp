#include "utils.h"

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

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
Status ReadLabelsFile(const string& file_name, std::vector<string>* result,
                      size_t* found_label_count) {
    std::ifstream file(file_name);
    if (!file) {
        return tensorflow::errors::NotFound("Labels file ", file_name,
                                            " not found.");
    }
    result->clear();
    string line;
    while (std::getline(file, line)) {
        result->push_back(line);
    }
    *found_label_count = result->size();
    const int padding = 16;
    while (result->size() % padding) {
        result->emplace_back();
    }
    return Status::OK();
}

static Status ReadEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output) {
    tensorflow::uint64 file_size = 0;
    TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

    string contents;
    contents.resize(file_size);

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

    tensorflow::StringPiece data;
    TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
    if (data.size() != file_size) {
        return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                            "' expected ", file_size, " got ",
                                            data.size());
    }
    output->scalar<string>()() = data.ToString();
    return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    string input_name = "file_reader";
    string output_name = "normalized";

    // read file_name into a tensor named input
    Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
    TF_RETURN_IF_ERROR(
            ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

    // use a placeholder to read input data
    auto file_reader =
            Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
            {"input", input},
    };

    // Now try to figure out what kind of file it is and decode it.
    const int wanted_channels = 3;
    tensorflow::Output image_reader;
    if (tensorflow::StringPiece(file_name).ends_with(".png")) {
        image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                                 DecodePng::Channels(wanted_channels));
    } else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
        // gif decoder returns 4-D tensor, remove the first dim
        image_reader =
                Squeeze(root.WithOpName("squeeze_first_dim"),
                        DecodeGif(root.WithOpName("gif_reader"), file_reader));
    } else {
        // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
        image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                                  DecodeJpeg::Channels(wanted_channels));
    }
    // Now cast the image data to float so we can do normal math on it.
    // auto float_caster =
    //     Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);

    auto uint8_caster =  Cast(root.WithOpName("uint8_caster"), image_reader, tensorflow::DT_UINT8);

    // The convention for image ops in TensorFlow is that all images are expected
    // to be in batches, so that they're four-dimensional arrays with indices of
    // [batch, height, width, channel]. Because we only have a single image, we
    // have to add a batch dimension of 1 to the start with ExpandDims().
    auto dims_expander = ExpandDims(root.WithOpName("dim"), uint8_caster, 0);

    // Bilinearly resize the image to fit the required dimensions.
    // auto resized = ResizeBilinear(
    //     root, dims_expander,
    //     Const(root.WithOpName("size"), {input_height, input_width}));


    // Subtract the mean and divide by the scale.
    // auto div =  Div(root.WithOpName(output_name), Sub(root, dims_expander, {input_mean}),
    //     {input_std});


    //cast to int
    //auto uint8_caster =  Cast(root.WithOpName("uint8_caster"), div, tensorflow::DT_UINT8);

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
            tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {"dim"}, {}, out_tensors));
    return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
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

Status readTensorFromMat(Mat mat, int input_height, int input_width, int input_depth, std::vector<Tensor>* out_tensors) {

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;

    tensorflow::TensorShape shape = tensorflow::TensorShape();
    shape.AddDim(input_height);
    shape.AddDim(input_width);
    shape.AddDim(input_depth);
    Tensor tensor(tensorflow::DT_FLOAT, shape);

    // use a placeholder to read input data
    auto input_tensor_mapped = tensor.tensor<float, 4>();

    auto matData = (float*) mat.data;
    for (int y = 0; y < input_height; ++y) {
        auto source_row = matData + (y * input_width * input_depth);
        for (int x = 0; x < input_width; ++x) {
            const float* source_pixel = source_row + (x * input_depth);
            for (int c = 0; c < input_depth; ++c) {
                const float* source_value = source_pixel + c;
                input_tensor_mapped(0, y, x, c) = *source_value;
            }
        }
    }
    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {{"input", tensor}};

    // Bilinearly resize the image to fit the required dimensions.
//     auto resized = ResizeBilinear(
//         root, dims_expander,
//         Const(root.WithOpName("size"), {input_height, input_width}));

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
            tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {"dim"}, {}, out_tensors));
    return Status::OK();
}