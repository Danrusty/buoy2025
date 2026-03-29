!> wdf_model_mod.f90
!! ==============================================================================
!! WDF_DL_Param 项目 Phase 3 - Fortran ONNX 推理接口
!!
!! 利用 iso_c_binding 调用 C++ ONNX Runtime 推理引擎。
!!
!! ★ 内存布局约定（关键）：
!!   Fortran = column-major，C++/ONNX = row-major。
!!   为使 Fortran 数组直接传给 C++ 时内存布局正确：
!!     input_features 声明为 (9, N)     → C++ 视为 row-major (N, 9) ✓
!!     output_uv      声明为 (2, N)     → C++ 视为 row-major (N, 2) ✓
!!
!!   在溢油模型中填充特征时，按如下方式：
!!     features(1, i) = era5_u10         ← 粒子 i 的第 1 个特征
!!     features(2, i) = era5_v10         ← 粒子 i 的第 2 个特征
!!     ...
!!     features(9, i) = era5_wave_dir_cos ← 粒子 i 的第 9 个特征
!!
!!   9 个特征的顺序（必须与 Python 训练时一致）：
!!     1: era5_u10           东向 10m 风速 (m/s)
!!     2: era5_v10           北向 10m 风速 (m/s)
!!     3: era5_wind_speed    风速标量 (m/s)
!!     4: era5_wind_dir_sin  sin(风向)
!!     5: era5_wind_dir_cos  cos(风向)
!!     6: era5_swh           有效波高 (m)
!!     7: era5_mwp           平均波周期 (s)
!!     8: era5_wave_dir_sin  sin(波向)
!!     9: era5_wave_dir_cos  cos(波向)
!!
!!   输出 (2, N)：
!!     output_uv(1, i) = residual_u  东向漂移残差 (m/s)
!!     output_uv(2, i) = residual_v  北向漂移残差 (m/s)
!! ==============================================================================
module wdf_model_mod
    use iso_c_binding
    implicit none
    private

    public :: wdf_init, wdf_predict, wdf_cleanup

    ! C 接口声明
    interface
        !> int init_onnx_model(const char* model_path);
        function init_onnx_model(model_path) bind(C, name="init_onnx_model")
            import :: c_char, c_int
            character(kind=c_char), intent(in) :: model_path(*)
            integer(c_int) :: init_onnx_model
        end function init_onnx_model

        !> int predict_drift(const float* input_features, int num_particles, float* output_uv);
        function predict_drift(input_features, num_particles, output_uv) &
                 bind(C, name="predict_drift")
            import :: c_float, c_int
            real(c_float), intent(in)  :: input_features(*)
            integer(c_int), value      :: num_particles
            real(c_float), intent(out) :: output_uv(*)
            integer(c_int) :: predict_drift
        end function predict_drift

        !> void cleanup_onnx();
        subroutine cleanup_onnx() bind(C, name="cleanup_onnx")
        end subroutine cleanup_onnx
    end interface

contains

    !> 初始化 ONNX 模型
    !! @param path  .onnx 模型文件路径
    !! @return success  是否成功
    function wdf_init(path) result(success)
        character(len=*), intent(in) :: path
        logical :: success
        character(kind=c_char, len=len_trim(path)+1) :: path_c
        integer(c_int) :: status

        path_c = trim(path) // c_null_char
        status = init_onnx_model(path_c)
        success = (status == 0)

        if (success) then
            print *, "[Fortran] ONNX model loaded: ", trim(path)
        else
            print *, "[Fortran ERROR] Failed to load ONNX model: ", trim(path)
        end if
    end function wdf_init

    !> 批量预测漂移残差
    !!
    !! @param features   输入特征数组，shape = (9, num_particles)
    !!                   ★ 第一维是特征，第二维是粒子编号
    !! @param np         粒子数量 N
    !! @param drift_uv   输出残差数组，shape = (2, num_particles)
    !!                   drift_uv(1,:) = residual_u, drift_uv(2,:) = residual_v
    subroutine wdf_predict(features, np, drift_uv)
        integer, intent(in)            :: np
        real(c_float), intent(in)      :: features(9, np)
        real(c_float), intent(out)     :: drift_uv(2, np)
        integer(c_int) :: status

        status = predict_drift(features, int(np, c_int), drift_uv)

        if (status /= 0) then
            print *, "[Fortran ERROR] WDF inference failed, status: ", status
        end if
    end subroutine wdf_predict

    !> 释放 ONNX Runtime 资源
    subroutine wdf_cleanup()
        call cleanup_onnx()
    end subroutine wdf_cleanup

end module wdf_model_mod
