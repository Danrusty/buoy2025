@echo off
REM ==========================================================================
REM build_wrapper.bat
REM 在 VS2022 Developer Command Prompt (x64) 中运行
REM 将 onnx_wrapper.cpp 编译为 wdf_onnx.dll + wdf_onnx.lib
REM ==========================================================================

REM --- 配置 ONNX Runtime 路径（按实际解压位置修改） ---
set ORT_DIR=D:\OilspillModel\OilSpillModel\onnxruntime-training-win-x64-1.17.1

REM --- 检查 ---
if not exist "%ORT_DIR%\include\onnxruntime_cxx_api.h" (
    echo [ERROR] 未找到 ONNX Runtime 头文件！
    echo         请确认 ORT_DIR=%ORT_DIR% 路径正确。
    echo         下载地址: https://github.com/microsoft/onnxruntime/releases
    echo         选择 onnxruntime-win-x64-1.17.x.zip
    exit /b 1
)

if not exist "%ORT_DIR%\lib\onnxruntime.lib" (
    echo [ERROR] 未找到 onnxruntime.lib！
    echo         请确认 %ORT_DIR%\lib\ 目录下有 onnxruntime.lib
    exit /b 1
)

echo.
echo === 编译 onnx_wrapper.cpp → wdf_onnx.dll ===
echo     ORT_DIR = %ORT_DIR%
echo.

cl /LD /EHsc /O2 /MD ^
   /I"%ORT_DIR%\include" ^
   onnx_wrapper.cpp ^
   /link /LIBPATH:"%ORT_DIR%\lib" onnxruntime.lib ^
   /OUT:wdf_onnx.dll /IMPLIB:wdf_onnx.lib

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] 编译失败！请检查错误信息。
    exit /b 1
)

echo.
echo === 编译成功 ===
echo 生成文件:
echo   wdf_onnx.dll   (运行时动态库)
echo   wdf_onnx.lib   (链接时导入库)
echo   wdf_onnx.exp   (导出文件，可忽略)
echo.
echo 下一步:
echo   1. 将 wdf_onnx.dll, wdf_onnx.lib 复制到溢油模型项目目录
echo   2. 将 %ORT_DIR%\lib\onnxruntime.dll 复制到同一目录
echo   3. 将 wdf_drifter.onnx 复制到运行目录
echo   4. 在 Fortran 项目中链接 wdf_onnx.lib 和 onnxruntime.lib
echo.
