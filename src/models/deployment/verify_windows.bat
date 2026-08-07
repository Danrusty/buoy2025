@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

if not defined VSDEVCMD (
    set "VSDEVCMD=C:\PROGRA~1\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
)
if not defined ONEAPI_SETVARS (
    set "ONEAPI_SETVARS=C:\PROGRA~2\Intel\oneAPI\setvars.bat"
)
if not defined ORT_DIR (
    set "ORT_DIR=D:\OilspillModel\OilSpillModel\onnxruntime-training-win-x64-1.17.1"
)

if not exist "%VSDEVCMD%" (
    echo [ERROR] VS2022 environment script not found: %VSDEVCMD%
    goto :fail
)
if not exist "%ONEAPI_SETVARS%" (
    echo [ERROR] oneAPI environment script not found: %ONEAPI_SETVARS%
    goto :fail
)
if not exist "%ORT_DIR%\lib\onnxruntime.dll" (
    echo [ERROR] ONNX Runtime DLL not found under: %ORT_DIR%
    goto :fail
)

call "%VSDEVCMD%" -arch=x64
if errorlevel 1 goto :fail
call "%ONEAPI_SETVARS%" intel64
if errorlevel 1 goto :fail

where cl >nul 2>&1
if errorlevel 1 (
    echo [ERROR] cl.exe is unavailable after environment setup.
    goto :fail
)
where ifx >nul 2>&1
if errorlevel 1 (
    echo [ERROR] ifx.exe is unavailable after environment setup.
    goto :fail
)

call build_wrapper.bat
if errorlevel 1 goto :fail

copy /Y "%ORT_DIR%\lib\onnxruntime.dll" ".\onnxruntime.dll" >nul

ifx /nologo /c /O2 /module:. /object:wdf_model_mod.obj wdf_model_mod.f90
if errorlevel 1 goto :fail

ifx /nologo /O2 /I. /exe:test_wdf_onnx.exe ^
    test_wdf_onnx.f90 wdf_model_mod.obj wdf_onnx.lib ^
    "%ORT_DIR%\lib\onnxruntime.lib"
if errorlevel 1 goto :fail

set "PATH=%SCRIPT_DIR%;%ORT_DIR%\lib;%PATH%"
test_wdf_onnx.exe
if errorlevel 1 goto :fail

echo.
echo [PASS] core6 C++/Fortran/ONNX acceptance test completed.
popd
endlocal
exit /b 0

:fail
echo.
echo [FAIL] Windows acceptance test failed.
popd
endlocal
exit /b 1
