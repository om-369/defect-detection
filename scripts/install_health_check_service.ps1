# PowerShell script to install the health check service
$serviceName = "DefectDetectionHealth"
$displayName = "Defect Detection Health Check Service"
$binPath = "python.exe $PSScriptRoot\health_check.py"
$startupType = "Automatic"

# Check if the service already exists
$service = Get-Service -Name $serviceName -ErrorAction SilentlyContinue

if ($service -eq $null) {
    # Create the service
    New-Service -Name $serviceName `
                -DisplayName $displayName `
                -BinaryPathName $binPath `
                -StartupType $startupType `
                -Description "Monitors the health of the Defect Detection service"

    Write-Host "Service installed successfully"
} else {
    Write-Host "Service already exists"
}

# Start the service
Start-Service -Name $serviceName
Write-Host "Service started"
