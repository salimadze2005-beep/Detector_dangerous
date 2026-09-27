[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PythonPathFile,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$PythonVersion = "3.12.10"
$PythonUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"
$VCRuntimeUrl = "https://aka.ms/vc14/vc_redist.x64.exe"
$CacheRoot = Join-Path $env:LOCALAPPDATA "DetectorDanger\installers"


function Write-Utf8NoBom([string]$Path, [string]$Value) {
    $parent = Split-Path -Parent $Path
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    [IO.File]::WriteAllText($Path, $Value, (New-Object Text.UTF8Encoding($false)))
}


function Test-Python312([string]$Path) {
    if (-not $Path -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    try {
        $result = & $Path -c "import struct,sys; print('OK' if sys.version_info[:2] == (3,12) and struct.calcsize('P')*8 == 64 else 'NO')" 2>$null
        return $LASTEXITCODE -eq 0 -and ($result | Select-Object -Last 1) -eq "OK"
    }
    catch {
        return $false
    }
}


function Find-Python312 {
    $candidates = New-Object Collections.Generic.List[string]
    if ($env:DETECTOR_PYTHON) {
        $candidates.Add($env:DETECTOR_PYTHON)
    }

    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        try {
            $launched = & $launcher.Source -3.12 -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $launched) {
                $candidates.Add(($launched | Select-Object -Last 1))
            }
        }
        catch { }
    }

    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $candidates.Add($pythonCommand.Source)
    }
    $candidates.Add((Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"))
    $candidates.Add("C:\Python312\python.exe")
    $candidates.Add("C:\Program Files\Python312\python.exe")

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-Python312 $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return $null
}


function Get-TrustedInstaller([string]$Url, [string]$Destination, [string]$PublisherPattern) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            Write-Host "[DOWNLOAD] $Url (attempt $attempt/5)"
            Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Destination -TimeoutSec 180
            $signature = Get-AuthenticodeSignature -FilePath $Destination
            if ($signature.Status -ne "Valid" -or
                -not $signature.SignerCertificate.Subject.Contains($PublisherPattern)) {
                throw "Invalid Authenticode signature: $($signature.Status) / $($signature.SignerCertificate.Subject)"
            }
            return
        }
        catch {
            if ($attempt -eq 5) { throw }
            Write-Warning "Download failed: $($_.Exception.Message). Retrying..."
            Start-Sleep -Seconds ([Math]::Min([Math]::Pow(2, $attempt), 15))
        }
    }
}


function Test-VCRuntime {
    foreach ($key in @(
        "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
    )) {
        try {
            $runtime = Get-ItemProperty -LiteralPath $key -ErrorAction Stop
            if ([int]$runtime.Installed -eq 1 -and [int]$runtime.Major -ge 14) {
                return $true
            }
        }
        catch { }
    }
    return $false
}


function Install-VCRuntime {
    if (Test-VCRuntime) {
        Write-Host "[OK] Microsoft Visual C++ Redistributable x64 is installed."
        return
    }
    if ($CheckOnly) {
        throw "Microsoft Visual C++ Redistributable x64 is missing"
    }

    $installer = Join-Path $CacheRoot "vc_redist.x64.exe"
    Get-TrustedInstaller $VCRuntimeUrl $installer "Microsoft Corporation"
    Write-Host "[INSTALL] Microsoft Visual C++ Redistributable x64 (Windows may ask for permission)..."
    $arguments = "/install /quiet /norestart"
    try {
        if (([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
            [Security.Principal.WindowsBuiltInRole]::Administrator
        )) {
            $process = Start-Process -FilePath $installer -ArgumentList $arguments -Wait -PassThru
        }
        else {
            $process = Start-Process -FilePath $installer -ArgumentList $arguments -Verb RunAs -Wait -PassThru
        }
    }
    catch {
        throw "Visual C++ runtime installation was cancelled or failed: $($_.Exception.Message)"
    }
    if ($process.ExitCode -notin @(0, 1638, 3010)) {
        throw "Visual C++ runtime installer returned exit code $($process.ExitCode)"
    }
    if (-not (Test-VCRuntime)) {
        throw "Visual C++ runtime is still not detected after installation"
    }
    Write-Host "[OK] Microsoft Visual C++ Redistributable x64 is ready."
}


try {
    if ([Environment]::Is64BitOperatingSystem -ne $true) {
        throw "Detector Danger requires 64-bit Windows 10/11"
    }

    Install-VCRuntime
    $python = Find-Python312
    if (-not $python) {
        if ($CheckOnly) {
            throw "Python 3.12 x64 is missing"
        }
        $installer = Join-Path $CacheRoot "python-$PythonVersion-amd64.exe"
        Get-TrustedInstaller $PythonUrl $installer "Python Software Foundation"
        Write-Host "[INSTALL] Python $PythonVersion x64 for the current user..."
        $target = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312"
        $arguments = @(
            "/quiet",
            "InstallAllUsers=0",
            "TargetDir=`"$target`"",
            "Include_pip=1",
            "Include_launcher=1",
            "InstallLauncherAllUsers=0",
            "Include_test=0",
            "PrependPath=1",
            "Shortcuts=0"
        ) -join " "
        $process = Start-Process -FilePath $installer -ArgumentList $arguments -Wait -PassThru
        if ($process.ExitCode -notin @(0, 3010)) {
            throw "Python installer returned exit code $($process.ExitCode)"
        }
        $python = Find-Python312
    }

    if (-not $python) {
        throw "Python 3.12 x64 was not found after installation"
    }
    Write-Utf8NoBom $PythonPathFile $python
    Write-Host "[OK] Python runtime: $python"
    exit 0
}
catch {
    Write-Error $_.Exception.Message
    exit 1
}
