!> test_wdf_onnx.f90
!! ==============================================================================
!! 端到端验证：Fortran → C++ → ONNX Runtime 推理链路
!!
!! 构造 3 组与 Python export_onnx.py REFERENCE_INPUTS 相同的输入，
!! 对比 Fortran 侧输出与 Python 参考输出是否一致。
!!
!! 编译方式（VS2022 Developer Command Prompt + Intel Fortran）：
!!   ifx /c wdf_model_mod.f90
!!   ifx test_wdf_onnx.f90 wdf_model_mod.obj wdf_onnx.lib onnxruntime.lib
!!   test_wdf_onnx.exe
!! ==============================================================================
program test_wdf_onnx
    use iso_c_binding
    use wdf_model_mod
    implicit none

    integer, parameter :: N = 3       ! 测试粒子数
    integer, parameter :: NF = 9      ! 特征数
    real(c_float), parameter :: TOL = 1.0e-4_c_float
    real(c_float), parameter :: EXPECTED(2, N) = reshape((/ &
         0.06901634_c_float,  0.01871330_c_float, &
        -0.06357081_c_float,  0.03777863_c_float, &
         0.01102608_c_float, -0.09855364_c_float  &
    /), (/ 2, N /))
    real(c_float) :: features(NF, N)  ! (9, 3) — 详见 wdf_model_mod 注释
    real(c_float) :: drift_uv(2, N)   ! (2, 3)
    real(c_float) :: max_error
    logical :: ok
    integer :: i

    ! ------------------------------------------------------------------
    ! 固定测试向量使用合理的物理量级，且风向编码与 u/v 自洽。
    ! ------------------------------------------------------------------
    ! 粒子 1
    features(1,1) =  5.0; features(2,1) =  0.0; features(3,1) =  5.0
    features(4,1) =  0.0; features(5,1) =  1.0; features(6,1) =  1.5
    features(7,1) =  7.0; features(8,1) =  0.0; features(9,1) =  1.0

    ! 粒子 2
    features(1,2) = -4.0; features(2,2) =  3.0; features(3,2) =  5.0
    features(4,2) =  0.6; features(5,2) = -0.8; features(6,2) =  2.5
    features(7,2) =  9.0; features(8,2) =  1.0; features(9,2) =  0.0

    ! 粒子 3
    features(1,3) =  0.0; features(2,3) = -8.0; features(3,3) =  8.0
    features(4,3) = -1.0; features(5,3) =  0.0; features(6,3) =  4.0
    features(7,3) = 12.0
    features(8,3) = -0.70710677; features(9,3) = 0.70710677

    ! ------------------------------------------------------------------
    ! 初始化模型（路径需根据实际调整）
    ! ------------------------------------------------------------------
    ok = wdf_init('wdf_drifter.onnx')
    if (.not. ok) then
        print *, "Model load failed, aborting test."
        stop 1
    end if

    ! ------------------------------------------------------------------
    ! 推理
    ! ------------------------------------------------------------------
    call wdf_predict(features, N, drift_uv)

    ! ------------------------------------------------------------------
    ! 输出结果
    ! ------------------------------------------------------------------
    print *, ""
    print *, "=== Fortran ONNX Inference Result ==="
    print *, "Particle   residual_u    residual_v"
    print *, "--------   ----------    ----------"
    do i = 1, N
        write(*, '(A, I2, A, F12.6, A, F12.6)') &
            "  #", i, "  ", drift_uv(1, i), "  ", drift_uv(2, i)
    end do
    print *, ""
    print *, "Python reference:"
    print *, "  #1   +0.069016    +0.018713"
    print *, "  #2   -0.063571    +0.037779"
    print *, "  #3   +0.011026    -0.098554"
    print *, ""
    max_error = maxval(abs(drift_uv - EXPECTED))
    write(*, '(A, ES12.4)') "Maximum absolute error: ", max_error
    if (max_error >= TOL) then
        print *, "Verification FAILED."
        call wdf_cleanup()
        stop 2
    end if
    print *, "Verification PASSED."

    ! ------------------------------------------------------------------
    ! 清理
    ! ------------------------------------------------------------------
    call wdf_cleanup()

end program test_wdf_onnx
