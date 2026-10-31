@echo off
REM Build the native cuBQL baseline. Requires:
REM   git clone --depth 1 https://github.com/NVIDIA/cuBQL ..\out\vendor\cuBQL
setlocal
if "%CUDACXX%"=="" set "CUDACXX=D:\CUDA\bin\nvcc.exe"
set "VS=C:\Program Files\Microsoft Visual Studio\2022\Community"
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"

"%CUDACXX%" -O3 -std=c++17 -arch=sm_89 -DNDEBUG -Xcompiler /utf-8 ^
   -I "..\out\vendor\cuBQL" cubql_bench.cu -o cubql_bench.exe || exit /b 1

echo.
echo built: %~dp0cubql_bench.exe
