# unblock.ps1: Strip Windows Mark-of-the-Web from TEMseg files
#
# When Windows downloads a file from the internet, it tags it with an alternate
# NTFS data stream called Zone.Identifier. If you unzip TEMseg with Explorer,
# that tag propagates to every file inside. .NET Framework refuses to load
# DLLs that carry this tag, which causes Python.Runtime.dll and other
# dependencies to fail on launch.
#
# Usage (run inside the extracted TEMseg folder):
#   powershell -ExecutionPolicy Bypass -File .\unblock.ps1
#

$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$target = Split-Path -Parent $here

Write-Host "Unblocking files in: $target"
Write-Host "This may take a few seconds..."

$files = Get-ChildItem -Path $target -Recurse -File
$count = 0
foreach ($file in $files) {
    Unblock-File -Path $file.FullName
    $count++
}

Write-Host "Done! unblocked $count files."
Write-Host "You can now launch TEMseg.exe."
