@echo off
REM Build the Embree baseline against the exact same scene/ray generators as quarry.
setlocal
if "%EMBREE_ROOT%"=="" set "EMBREE_ROOT=D:\code\quarry-bench\embree-4.4.1.x64.windows"
set "VS=C:\Program Files\Microsoft Visual Studio\2022\Community"
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"

cl /nologo /O2 /EHsc /std:c++17 /utf-8 /W4 /I "..\src" /I "%EMBREE_ROOT%\include" ^
   embree_bench.cpp ..\src\core\scene.cpp ..\src\cpu\brute.cpp ^
   /link /LIBPATH:"%EMBREE_ROOT%\lib" embree4.lib /OUT:embree_bench.exe || exit /b 1

if not exist embree4.dll copy /y "%EMBREE_ROOT%\bin\embree4.dll" . >nul
if not exist tbb12.dll copy /y "%EMBREE_ROOT%\bin\tbb12.dll" . >nul
if not exist tbbmalloc.dll copy /y "%EMBREE_ROOT%\bin\tbbmalloc.dll" . >nul

echo.
echo built: %~dp0embree_bench.exe
