# setup_kernel.ps1
# Crea el venv, instala dependencias y registra el kernel de Jupyter
# Ejecutar desde la carpeta del proyecto:
#   cd "C:\Users\rodri\Desktop\U\2026\2026-1\Deep web\proyecto_deep"
#   powershell -ExecutionPolicy Bypass -File setup_kernel.ps1

$ErrorActionPreference = "Stop"
$VENV = ".venv"
$KERNEL_NAME = "pothole_deep"
$DISPLAY_NAME = "Python (pothole_deep)"

Write-Host "`n[1/4] Creando entorno virtual..." -ForegroundColor Cyan
python -m venv $VENV

Write-Host "[2/4] Actualizando pip..." -ForegroundColor Cyan
& "$VENV\Scripts\python.exe" -m pip install --upgrade pip

Write-Host "[3/4] Instalando dependencias..." -ForegroundColor Cyan
& "$VENV\Scripts\pip.exe" install -r requirements.txt
& "$VENV\Scripts\pip.exe" install ipykernel jupyter python-dotenv

Write-Host "[4/4] Registrando kernel en Jupyter..." -ForegroundColor Cyan
& "$VENV\Scripts\python.exe" -m ipykernel install --user --name $KERNEL_NAME --display-name $DISPLAY_NAME

Write-Host "`nListo. Kernel registrado como: '$DISPLAY_NAME'" -ForegroundColor Green
Write-Host "Abre el notebook y selecciona el kernel: $DISPLAY_NAME`n" -ForegroundColor Green
