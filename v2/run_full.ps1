# Full 14B labeling run: submissions, then comment batch 1, then merge-validate and tables.
# Resumable: run this script again and finished ids are skipped (jsonl caches).
$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
Set-Location -LiteralPath $PSScriptRoot
$py = "C:\Users\Daisy\anaconda3\envs\evllm\python.exe"
$ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
$log = Join-Path $PSScriptRoot "outputs\full_run.log"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content -LiteralPath $log -Encoding utf8 }
function Ensure-Ollama {
  try { Invoke-RestMethod http://127.0.0.1:11434/api/version -TimeoutSec 5 | Out-Null }
  catch { Log "ollama not responding; starting ollama serve"; Start-Process $ollama -ArgumentList "serve" -WindowStyle Hidden; Start-Sleep 15 }
}
function Run-Label($label, [string[]]$a) {
  for ($try = 1; $try -le 20; $try++) {
    Ensure-Ollama
    Log "START $label (attempt $try)"
    & $py @a 2>&1 | ForEach-Object { Log "$_" }
    $code = $LASTEXITCODE
    Log "END $label exit=$code"
    if ($code -eq 0) { return $true }
    Start-Sleep 60
  }
  return $false
}
Log "=== run_full.ps1 started pid=$PID ==="
$common = @("llm_label.py","--backend","ollama","--base-url","http://127.0.0.1:11434/v1","--num-ctx","4096","--batch-size","1")
$ok1 = Run-Label "submissions" ($common + @("--posts","data/raw/label_posts.csv","--out","outputs/llm_labels_submissions.csv","--cache","data/raw/llm_labels_submissions.jsonl"))
$ok2 = Run-Label "comments" ($common + @("--posts","data/raw/comment_label_batch1_posts.csv","--out","outputs/llm_labels_comments.csv","--cache","data/raw/llm_labels_comments.jsonl"))
if ($ok1 -and $ok2) {
  Log "START merge-validate"
  & $py llm_label.py --merge-validate --expect data/raw/label_posts.csv --expect data/raw/comment_label_batch1_posts.csv --caches data/raw/llm_labels_submissions.jsonl --caches data/raw/llm_labels_comments.jsonl --out outputs/llm_labels.csv 2>&1 | ForEach-Object { Log "$_" }
  $m = $LASTEXITCODE; Log "END merge-validate exit=$m"
  if ($m -eq 0) {
    Log "START run_after_llm"
    & $py run_after_llm.py --llm outputs/llm_labels.csv --out outputs/after_llm --n-boot 1000 2>&1 | ForEach-Object { Log "$_" }
    Log "END run_after_llm exit=$LASTEXITCODE"
  }
} else { Log "labeling did not finish cleanly; rerun run_full.ps1 to resume" }
Log "=== run_full.ps1 finished ==="
