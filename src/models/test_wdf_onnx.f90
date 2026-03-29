!> test_wdf_onnx.f90
!! ==============================================================================
!! 端到端验证：Fortran → C++ → ONNX Runtime 推理链路
!!
!! 构造 3 组与 Python export_onnx.py verify() 相同的输入，
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
    real(c_float) :: features(NF, N)  ! (9, 3) — 详见 wdf_model_mod 注释
    real(c_float) :: drift_uv(2, N)   ! (2, 3)
    logical :: ok
    integer :: i

    ! ------------------------------------------------------------------
    ! Python export_onnx.py verify() 使用 np.random.seed(42) 生成的
    ! 前 3 组 × 9 特征的值（np.random.randn(100,9)*5.0 的前 3 行）
    ! ------------------------------------------------------------------
    ! 粒子 1 (np.random.seed(42), randn(3,9)*5.0, row 0)
    features(1,1) =  2.4836; features(2,1) = -0.6913; features(3,1) =  3.2384
    features(4,1) =  7.6151; features(5,1) = -1.1708; features(6,1) = -1.1707
    features(7,1) =  7.8961; features(8,1) =  3.8372; features(9,1) = -2.3474

    ! 粒子 2 (row 1)
    features(1,2) =  2.7128; features(2,2) = -2.3171; features(3,2) = -2.3286
    features(4,2) =  1.2098; features(5,2) = -9.5664; features(6,2) = -8.6246
    features(7,2) = -2.8114; features(8,2) = -5.0642; features(9,2) =  1.5712

    ! 粒子 3 (row 2)
    features(1,3) = -4.5401; features(2,3) = -7.0615; features(3,3) =  7.3282
    features(4,3) = -1.1289; features(5,3) =  0.3376; features(6,3) = -7.1237
    features(7,3) = -2.7219; features(8,3) =  0.5546; features(9,3) = -5.7550

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
    print *, "  #1   -0.438177    +0.068159"
    print *, "  #2   -0.288603    +0.472317"
    print *, "  #3   -0.737593    +0.216485"
    print *, ""
    print *, "If diff < 1e-4, verification PASSED."

    ! ------------------------------------------------------------------
    ! 清理
    ! ------------------------------------------------------------------
    call wdf_cleanup()

end program test_wdf_onnx
