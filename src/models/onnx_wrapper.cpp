/**
 * onnx_wrapper.cpp
 * ================
 * WDF_DL_Param Phase 3 - Task B
 *
 * C-style wrapper for ONNX Runtime C++ API.
 * Called from Fortran via iso_c_binding.
 */

#ifdef _WIN32
#include <windows.h>
#endif
#include "onnx_wrapper.h"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <memory>
#include <string>

// --- Global static variables ---
static std::unique_ptr<Ort::Env> g_env = nullptr;
static std::unique_ptr<Ort::Session> g_session = nullptr;
static std::unique_ptr<Ort::MemoryInfo> g_memory_info = nullptr;

// Model metadata (must match export_onnx.py)
static const char* INPUT_NAME = "input";
static const char* OUTPUT_NAME = "output";
static const int64_t INPUT_DIM = 9;
static const int64_t OUTPUT_DIM = 2;

extern "C" {

int init_onnx_model(const char* model_path) {
    try {
        g_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "WDF_Drift_Model");

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Windows ONNX Runtime requires wchar_t* path
#ifdef _WIN32
        int len = MultiByteToWideChar(CP_UTF8, 0, model_path, -1, nullptr, 0);
        std::wstring wpath(len, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, model_path, -1, &wpath[0], len);
        g_session = std::make_unique<Ort::Session>(*g_env, wpath.c_str(), session_options);
#else
        g_session = std::make_unique<Ort::Session>(*g_env, model_path, session_options);
#endif

        g_memory_info = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault));

        std::cout << "[ONNX] Model loaded: " << model_path << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ONNX] Failed to load model: " << e.what() << std::endl;
        return -1;
    }
}

int predict_drift(const float* input_features, int num_particles, float* output_uv) {
    if (!g_session) {
        std::cerr << "[ONNX] Error: session not initialized. Call init_onnx_model first." << std::endl;
        return -1;
    }

    try {
        std::vector<int64_t> input_shape = { static_cast<int64_t>(num_particles), INPUT_DIM };

        size_t input_count = num_particles * INPUT_DIM;
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            *g_memory_info,
            const_cast<float*>(input_features),
            input_count,
            input_shape.data(),
            input_shape.size()
        );

        const char* input_names[] = { INPUT_NAME };
        const char* output_names[] = { OUTPUT_NAME };

        auto output_tensors = g_session->Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        float* result_ptr = output_tensors.front().GetTensorMutableData<float>();
        size_t result_count = num_particles * OUTPUT_DIM;
        std::copy(result_ptr, result_ptr + result_count, output_uv);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ONNX] Inference error: " << e.what() << std::endl;
        return -2;
    }
}

void cleanup_onnx() {
    g_session.reset();
    g_env.reset();
    g_memory_info.reset();
    std::cout << "[ONNX] Resources released." << std::endl;
}

} // extern "C"
