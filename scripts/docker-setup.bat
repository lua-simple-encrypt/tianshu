@echo off
REM Tianshu - Enterprise AI Data Preprocessing Platform
REM Docker Quick Setup Script for Windows (WSL2 Optimized)

setlocal enabledelayedexpansion

REM ============================================================================
REM Configuration
REM ============================================================================
set "PROJECT_NAME=tianshu"
set "DOCKER_BUILDKIT=1"
set "COMPOSE_DOCKER_CLI_BUILD=1"

REM Navigate to project root (Parent directory of scripts/)
cd /d "%~dp0\.."

:main_entry
cls
echo ======================================================================
echo    Tianshu - Enterprise AI Preprocessing Platform (Windows/WSL2)
echo ======================================================================
echo.

REM ============================================================================
REM System Checks
REM ============================================================================
:check_requirements
echo [CHECK] Checking Docker environment...

REM 1. Check Docker Daemon
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running!
    echo [INFO]  Please start Docker Desktop and ensure WSL2 integration is enabled.
    pause
    exit /b 1
)

REM 2. Determine Compose Command (V2 vs V1)
docker compose version >nul 2>&1
if not errorlevel 1 (
    set "COMPOSE_CMD=docker compose"
) else (
    docker-compose --version >nul 2>&1
    if not errorlevel 1 (
        set "COMPOSE_CMD=docker-compose"
    ) else (
        echo [ERROR] Docker Compose not found. Please update Docker Desktop.
        pause
        exit /b 1
    )
)

REM 3. GPU Detection (Host Side)
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo [INFO]  NVIDIA GPU detected on host.
    echo         Note: Ensure "Use the WSL 2 based engine" is checked in Docker Desktop settings.
) else (
    echo [WARN]  NVIDIA GPU not detected on host. Setup will run in CPU-only mode.
)

echo [OK]    Environment ready. Using: !COMPOSE_CMD!
echo.

REM ============================================================================
REM Main Menu
REM ============================================================================
:menu
echo ======================================================================
echo    Select Operation
echo ======================================================================
echo.
echo    1. Full Setup (Initialize + Build + Start)
echo    2. Build/Rebuild Images Only (Recommended for first run)
echo    3. Start Services (Background)
echo    4. Show Live Logs
echo    5. Restart Services
echo    6. Stop Services
echo    7. System Prune (Clean Unused Data)
echo    8. Clean All Data (Factory Reset)
echo    0. Exit
echo.
set /p choice="Enter option [0-8]: "

if "%choice%"=="1" goto op_full_setup
if "%choice%"=="2" goto op_build
if "%choice%"=="3" goto op_start
if "%choice%"=="4" goto op_logs
if "%choice%"=="5" goto op_restart
if "%choice%"=="6" goto op_stop
if "%choice%"=="7" goto op_prune
if "%choice%"=="8" goto op_clean
if "%choice%"=="0" goto end
goto menu

REM ============================================================================
REM 1. Full Setup
REM ============================================================================
:op_full_setup
echo.
echo [INFO] Starting full setup process...

REM 1.1 Environment File
if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo [OK]   Created .env file from template.
        echo [WARN] Please edit .env later to set secure passwords!
    ) else (
        echo [ERROR] .env.example missing! Cannot proceed.
        pause
        goto menu
    )
) else (
    echo [SKIP] .env already exists.
)

REM 1.2 Directories
echo [INFO] Creating directory structure...
if not exist models mkdir models
if not exist data\uploads mkdir data\uploads
if not exist data\output mkdir data\output
if not exist data\db mkdir data\db
if not exist logs\backend mkdir logs\backend
if not exist logs\worker mkdir logs\worker
if not exist logs\mcp mkdir logs\mcp

REM 1.3 Build
echo.
echo [INFO] Building images (BuildKit Enabled)...
echo        This may take 10-20 mins for the first time due to CUDA libraries.
!COMPOSE_CMD! build
if errorlevel 1 (
    echo [ERROR] Build failed. Please check the error messages above.
    pause
    goto menu
)

REM 1.4 Start
echo.
echo [INFO] Starting services...
!COMPOSE_CMD! up -d
if errorlevel 1 (
    echo [ERROR] Startup failed.
    pause
    goto menu
)

goto show_success

REM ============================================================================
REM 2. Build Only
REM ============================================================================
:op_build
echo.
echo [INFO] Starting build process...
echo [INFO] Enabling BuildKit for faster caching...
set DOCKER_BUILDKIT=1
!COMPOSE_CMD! build --progress=plain
if errorlevel 1 (
    echo [ERROR] Build failed.
    pause
) else (
    echo [OK]    Build completed successfully.
    pause
)
goto menu

REM ============================================================================
REM 3. Start
REM ============================================================================
:op_start
echo.
echo [INFO] Starting existing containers...
!COMPOSE_CMD! up -d
goto show_success

REM ============================================================================
REM 4. Logs
REM ============================================================================
:op_logs
echo.
echo [INFO] Attaching to logs (Press Ctrl+C to detach)...
!COMPOSE_CMD! logs -f
goto menu

REM ============================================================================
REM 5. Restart
REM ============================================================================
:op_restart
echo.
echo [INFO] Restarting all services...
!COMPOSE_CMD! restart
echo [OK]    Services restarted.
pause
goto menu

REM ============================================================================
REM 6. Stop
REM ============================================================================
:op_stop
echo.
echo [INFO] Stopping all services...
!COMPOSE_CMD! stop
echo [OK]    Services stopped.
pause
goto menu

REM ============================================================================
REM 7. Prune (Safe Clean)
REM ============================================================================
:op_prune
echo.
echo [WARN] This will remove stopped containers and dangling images.
echo        Your data (database, uploads) will NOT be deleted.
set /p confirm="Continue? (y/n): "
if /i "%confirm%"=="y" (
    docker system prune -f
    echo [OK] System pruned.
    pause
)
goto menu

REM ============================================================================
REM 8. Clean All (Dangerous)
REM ============================================================================
:op_clean
echo.
echo [DANGER] ========================================================
echo          This will DESTROY ALL DATA including:
echo          - Database (Users, Tasks)
echo          - Uploaded Files
echo          - Processing Outputs
echo          - Logs
echo          ========================================================
set /p confirm="Type 'yes' to confirm strict cleanup: "
if /i not "%confirm%"=="yes" goto menu

echo [INFO] Stopping containers...
!COMPOSE_CMD! down -v

echo [INFO] Removing data directories...
REM Use ping as a delay to allow file handles to close
ping 127.0.0.1 -n 3 >nul

rmdir /s /q data 2>nul
if exist data (
    echo [WARN] Could not remove 'data' folder. Some files may be in use.
    echo        Please restart Docker Desktop and try again.
) else (
    echo [OK]   Data cleaned.
)

rmdir /s /q logs 2>nul
REM Note: We usually keep 'models' to avoid re-downloading large files
echo [INFO] Note: 'models' directory was preserved to save download time.
pause
goto menu

REM ============================================================================
REM Success Info
REM ============================================================================
:show_success
echo.
echo ======================================================================
echo    Setup Complete!
echo ======================================================================
echo.
echo    [Services]
echo    - API Server:      http://localhost:8000/docs
echo    - LitServe Worker: http://localhost:8001
echo    - MCP Server:      http://localhost:8002
echo.
echo    [Next Steps]
echo    1. Initial download of models (MinerU/PaddleOCR) will happen automatically.
echo       Check logs via Option 4 to see progress.
echo    2. The API might take 1-2 minutes to become responsive during initialization.
echo.
pause
goto menu

:end
endlocal
exit /b 0
