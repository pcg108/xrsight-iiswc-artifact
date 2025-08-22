#include "illixr/plugin.hpp"
#include "illixr/opencv_data_types.hpp"
#include "illixr/data_format.hpp"
#include "illixr/phonebook.hpp"
#include "illixr/eye_tracking_host.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>  
#include <opencv2/imgcodecs.hpp>  
#include <opencv2/core.hpp> 
#include <opencv2/core/mat.hpp>

#include <filesystem>
#include <shared_mutex>


using namespace ILLIXR;


static constexpr const int width_ = 240;
static constexpr const int height_ = 160;

class eye_tracking_host_impl : public eye_tracking_host {
public:
    explicit eye_tracking_host_impl(const phonebook* const pb)
        : sb{pb->lookup_impl<switchboard>()}
        , _m_clock{pb->lookup_impl<RelativeClock>()}
        { 
            env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ILLIXR_EyeTracking");
            session_options = Ort::SessionOptions();
            auto available_providers = Ort::GetAvailableProviders();
    
            std::cout << "Available providers: " << std::endl;
            for (const auto& provider : available_providers) {
                std::cout << "- " << provider << std::endl;
            }
    
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            std::string model_path = std::getenv("ILLIXR_EYE_MODEL");   
            if (model_path.empty()) {
                throw std::runtime_error("Model path is not set. Please set the ILLIXR_EYE_MODEL environment variable.");
            }

            session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
            input_tensor_ = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(),
                                            input_shape_.data(), input_shape_.size()));
            output_tensor_ = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(),
                                            output_shape_.data(), output_shape_.size()));
                
        }


    void print_first_5_rows(const cv::Mat& mat) {
        int rows_to_print = std::min(5, mat.rows);
        cv::Mat first_rows = mat(cv::Range(0, rows_to_print), cv::Range::all());
    
        std::cout << "First " << rows_to_print << " rows of matrix:\n" << first_rows << std::endl;
    }


    eye_position_type get_eye_position(cv::Mat img)  {

        const size_t total_elements = width_ * height_;
        if (img.total() != total_elements) {
            throw std::runtime_error("Dimension mismatch between img_scaled and input_image_");
        }
        std::memcpy(input_image_.data(), img.ptr<float>(), total_elements * sizeof(float));

        print_first_5_rows(img);

        const char* input_names[] = {"x"};
        const char* output_names[] = {"conv2d_41"};
        session->Run(run_options, input_names, input_tensor_.get(), 1, output_names, output_tensor_.get(), 1);

        float pred_x, pred_y;
        get_fovea(pred_y, pred_x);
        std::cout << "predicted fovea: " << pred_x << ", " << pred_y << std::endl; 

        return eye_position_type{_m_clock->now(), pred_x, pred_y};
    }


private:


    void get_fovea(float& fovea_x, float& fovea_y) {
        std::vector<uint8_t> argmax_map(height_ * width_);
        float* output_data = output_tensor_->GetTensorMutableData<float>();

        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                int max_class = 0;
                float max_value = output_data[0 * height_ * width_ + y * width_ + x];
                for (int c = 1; c < 4; ++c) {
                    float value = output_data[c * height_ * width_ + y * width_ + x];
                    if (value > max_value) {
                        max_value = value;
                        max_class = c;
                    }
                }
                argmax_map[y * width_ + x] = static_cast<uint8_t>(max_class);
            }
        }

        double sum_x = 0.0;
        double sum_y = 0.0;
        int count = 0;

        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                if (argmax_map[y * width_ + x] != 0) { // Assuming class 0 is background
                    sum_x += x;
                    sum_y += y;
                    ++count;
                }
            }
        }

        fovea_x = (count > 0) ? (sum_x / count) : 0.0;
        fovea_y = (count > 0) ? (sum_y / count) : 0.0;
    }


    const std::shared_ptr<switchboard>                               sb;
    const std::shared_ptr<const RelativeClock>                       _m_clock;
    
    Ort::Env env;
    Ort::SessionOptions session_options;
    OrtCUDAProviderOptions cuda_options;
    Ort::RunOptions run_options;
    std::unique_ptr<Ort::Value> input_tensor_;
    std::unique_ptr<Ort::Value> output_tensor_;
    std::unique_ptr<Ort::Session> session;

    std::array<int64_t, 4> input_shape_{1, 1, height_, width_};
    std::array<int64_t, 4> output_shape_{1, 4, height_, width_};
    std::array<float, width_ * height_> input_image_{};
    std::array<float, 4 * width_ * height_> results_{};
};

class eye_tracking_host_plugin : public plugin {
public:
    eye_tracking_host_plugin(const std::string& name, phonebook* pb)
        : plugin{name, pb} {
        pb->register_impl<eye_tracking_host>(
            std::static_pointer_cast<eye_tracking_host>(std::make_shared<eye_tracking_host_impl>(pb)));
    }
};

PLUGIN_MAIN(eye_tracking_host_plugin);