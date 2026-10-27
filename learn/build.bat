@echo off
REM Build single-file exercises in learn\. Usage: build.bat 01_bruteforce.cpp
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
if "%~1"=="" (
  for %%f in (*.cpp) do (
    echo [cl] %%f
    cl /nologo /O2 /EHsc /std:c++17 /utf-8 "%%f" /Fe:"%%~nf.exe" || exit /b 1
  )
  exit /b 0
)
cl /nologo /O2 /EHsc /std:c++17 /utf-8 "%~1" /Fe:"%~n1.exe" || exit /b 1
echo.
echo run: %~n1.exe
