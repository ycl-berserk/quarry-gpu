@echo off
REM Build with the CMake + Ninja bundled with Visual Studio 2022. No extra install needed.
setlocal
set "VS=C:\Program Files\Microsoft Visual Studio\2022\Community"
set "CMAKE=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "ROOT=%~dp0"

call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

"%CMAKE%" -S "%ROOT%." -B "%ROOT%build" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
  -DCMAKE_CXX_COMPILER=cl || exit /b 1

"%CMAKE%" --build "%ROOT%build" || exit /b 1

echo.
echo built: %ROOT%build\quarry.exe
echo try:   build\quarry.exe selftest
