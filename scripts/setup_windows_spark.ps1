# One-off setup for running the Spark pipeline locally on Windows.
#
# Spark's local file system writes go through Hadoop's native shim on
# Windows (winutils.exe and hadoop.dll). Linux, macOS, CI and Databricks
# need none of this; it is purely a Windows development convenience.
#
# Binaries come from the community-maintained cdarlint/winutils repository
# and are verified against pinned SHA-256 digests before being installed
# to %LOCALAPPDATA%\devtools\hadoop.
#
# After running this script, set the following for pipeline runs:
#   $env:HADOOP_HOME = "$env:LOCALAPPDATA\devtools\hadoop"
#   $env:Path = "$env:HADOOP_HOME\bin;$env:Path"

$ErrorActionPreference = 'Stop'

$hadoopBin = Join-Path $env:LOCALAPPDATA 'devtools\hadoop\bin'
$source = 'https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.6/bin'

$files = @(
    @{ Name = 'winutils.exe'; Sha256 = '496A591EB1E67DF2A620F710D529BA6DDFE1C19149E6647CC4E320BB0EFD8553' },
    @{ Name = 'hadoop.dll';   Sha256 = 'D7AB36A68518748CEF142BE2DA5069B4C763C2CD764C1D2E6AC48C7200405BE3' }
)

New-Item -ItemType Directory -Force $hadoopBin | Out-Null

foreach ($file in $files) {
    $target = Join-Path $hadoopBin $file.Name
    if (Test-Path $target) {
        Write-Output "$($file.Name) already present, skipping download"
    } else {
        Write-Output "Downloading $($file.Name)"
        curl.exe --fail --location --silent --show-error -o $target "$source/$($file.Name)"
    }
    $actual = (Get-FileHash $target -Algorithm SHA256).Hash
    if ($actual -ne $file.Sha256) {
        Remove-Item $target
        throw "$($file.Name) failed its checksum: expected $($file.Sha256), got $actual"
    }
    Write-Output "$($file.Name) verified"
}

Write-Output "Done. Set HADOOP_HOME to $(Split-Path $hadoopBin) before running the pipeline."
