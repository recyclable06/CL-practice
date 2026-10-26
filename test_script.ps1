Set-Location $PSScriptRoot

$python = ".\.venv\Scripts\python.exe"
$entry = ".\阶段3加参数.py"


$methods = @("finetune", "er", "derpp")
$seeds = @(1)

foreach($method in $methods){
    foreach($seed in $seeds){
        $output_path = ".\runs\run_${method}_${seed}"
        $runArgs = @(
            "--method", $method
            "--num_task", "3"
            "--num_epochs", "5"
            "--batch_size", "8"
            "--seed", $seed
            "--lr", 0.001
            "--cache_size", 200
            "--alpha", 0.5
            "--beta", 0.5
            "--output_dir", $output_path
        )
        New-Item -ItemType Directory -Force -Path $output_path | Out-Null
        & $python -u $entry @runArgs 2>&1 | Tee-Object $output_path\console.log

        if ($LASTEXITCODE -ne 0) {
            throw "训练失败：method=$method, seed=$seed"
        }
    }
}
Write-Host "全部实验完成."