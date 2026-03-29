#ifndef ONNX_WRAPPER_H
#define ONNX_WRAPPER_H

/**
 * onnx_wrapper.h
 * ==============
 * WDF_DL_Param 项目 Phase 3 - 任务 B
 * 
 * 这是一个 C 风格的包装头文件，旨在通过 Fortran 的 iso_c_binding 模块
 * 提供对 ONNX Runtime 推理引擎的访问。
 */

#ifdef _WIN32
  #define WDF_EXPORT __declspec(dllexport)
#else
  #define WDF_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 初始化 ONNX Runtime 环境和模型会话。
 */
WDF_EXPORT int init_onnx_model(const char* model_path);

/**
 * 执行批量推理。
 */
WDF_EXPORT int predict_drift(const float* input_features, int num_particles, float* output_uv);

/**
 * 释放 ONNX Runtime 资源，关闭会话。
 */
WDF_EXPORT void cleanup_onnx();

#ifdef __cplusplus
}
#endif

#endif // ONNX_WRAPPER_H
