@echo off
setlocal

if not defined ORT_DIR (
    set "ORT_DIR=D:\OilspillModel\OilSpillModel\onnxruntime-training-win-x64-1.17.1"
)

if not exist "%ORT_DIR%\include\onnxruntime_cxx_api.h" (
    echo [ERROR] ONNX Runtime header not found under: %ORT_DIR%
    endlocal
    exit /b 1
)
if not exist "%ORT_DIR%\lib\onnxruntime.lib" (
    echo [ERROR] onnxruntime.lib not found under: %ORT_DIR%
    endlocal
    exit /b 1
)

echo.
echo === Building x64 wdf_onnx.dll ===
echo ORT_DIR=%ORT_DIR%
echo.

cl /LD /EHsc /O2 /MD /utf-8 ^
   /I"%ORT_DIR%\include" ^
   onnx_wrapper.cpp ^
   /link /LIBPATH:"%ORT_DIR%\lib" onnxruntime.lib ^
   /OUT:wdf_onnx.dll /IMPLIB:wdf_onnx.lib

if errorlevel 1 (
    echo.
    echo [ERROR] C++ wrapper build failed.
    endlocal
    exit /b 1
)

echo.
echo [PASS] Generated wdf_onnx.dll and wdf_onnx.lib.
endlocal
exit /b 0
