param([string]$XeLaTeX = 'xelatex')
$ErrorActionPreference = 'Stop'
$paperRoot = $PSScriptRoot
$projectRoot = Split-Path (Split-Path $paperRoot -Parent) -Parent
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'pyproject.toml'))) {
    $projectRoot = $paperRoot
}
$buildRoot = Join-Path $projectRoot 'tmp/paper-build'
$outputRoot = Join-Path $projectRoot 'output/pdf'
New-Item -ItemType Directory -Force -Path $buildRoot, $outputRoot | Out-Null
$compiler = (Get-Command $XeLaTeX -ErrorAction Stop).Source
Push-Location $paperRoot
try {
    for ($pass = 1; $pass -le 3; $pass++) {
        & $compiler '-interaction=nonstopmode' '-halt-on-error' '-file-line-error' "-output-directory=$buildRoot" 'main.tex'
        if ($LASTEXITCODE -ne 0) { throw "XeLaTeX failed on pass $pass; inspect $buildRoot/main.log" }
    }
    $logText = Get-Content (Join-Path $buildRoot 'main.log') -Raw
    if ($logText -match 'undefined references|Citation .+ undefined|Missing character:') {
        throw 'Unresolved references, citations, or missing characters remain.'
    }
    $pdfPath = Join-Path $outputRoot 'rst19-starfield-paper.pdf'
    Copy-Item -LiteralPath (Join-Path $buildRoot 'main.pdf') -Destination $pdfPath -Force
    Write-Output "PDF: $pdfPath"
} finally { Pop-Location }
