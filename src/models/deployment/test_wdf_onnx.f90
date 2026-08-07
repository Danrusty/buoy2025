!> test_wdf_onnx.f90
!! ==============================================================================
!! 端到端验证：Fortran → C++ → ONNX Runtime 推理链路
!!
!! 从发布包 CSV 读取 Python 端固定输入与预期输出，
!! 对比 Fortran 侧输出与 Python ONNX Runtime 结果是否一致。
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
    integer, parameter :: NF = 6      ! core6 特征数
    real(c_float), parameter :: TOL = 1.0e-4_c_float
    real(c_float) :: features(NF, N)  ! (6, 3) — 详见 wdf_model_mod 注释
    real(c_float) :: expected(2, N)   ! Python ONNX Runtime 参考输出
    real(c_float) :: drift_uv(2, N)   ! (2, 3)
    real(c_float) :: max_error
    logical :: ok
    integer :: i, input_unit, output_unit, io_status
    character(len=512) :: header

    ! ------------------------------------------------------------------
    ! 从发布包读取固定向量，避免模型更新后保留旧的硬编码期望值。
    ! ------------------------------------------------------------------
    open(newunit=input_unit, file='test_input.csv', status='old', &
         action='read', iostat=io_status)
    if (io_status /= 0) then
        print *, "Failed to open test_input.csv."
        stop 1
    end if
    read(input_unit, '(A)', iostat=io_status) header
    do i = 1, N
        read(input_unit, *, iostat=io_status) features(:, i)
        if (io_status /= 0) then
            print *, "Failed to read test_input.csv row: ", i
            close(input_unit)
            stop 1
        end if
    end do
    close(input_unit)

    open(newunit=output_unit, file='expected_output.csv', status='old', &
         action='read', iostat=io_status)
    if (io_status /= 0) then
        print *, "Failed to open expected_output.csv."
        stop 1
    end if
    read(output_unit, '(A)', iostat=io_status) header
    do i = 1, N
        read(output_unit, *, iostat=io_status) expected(:, i)
        if (io_status /= 0) then
            print *, "Failed to read expected_output.csv row: ", i
            close(output_unit)
            stop 1
        end if
    end do
    close(output_unit)

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
    do i = 1, N
        write(*, '(A, I2, A, F12.6, A, F12.6)') &
            "  #", i, "  ", expected(1, i), "  ", expected(2, i)
    end do
    print *, ""
    max_error = maxval(abs(drift_uv - expected))
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
