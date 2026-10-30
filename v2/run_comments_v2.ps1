# Comments v2: label the balanced 8,000-comment draw with parent title in the prompt,
# then merge-validate and rerun after_llm as a sensitivity check (see COMMENTS_V2_PLAN.md).
# Resumable: run again and finished ids are skipped (data/raw/llm_labels_comments_v2.jsonl).
# Build the draw first with: python build_comments_v2.py
$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
Set-Location -LiteralPath $PSScriptRoot
$py = "C:\Users\Daisy\anaconda3\envs\evllm\python.exe"
$ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
$log = Join-Path $PSScriptRoot "outputs\comments_v2_run.log"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content -LiteralPath $log -Encoding utf8 }
function Ensure-Ollama {
  try { Invoke-RestMethod http://127.0.0.1:11434/api/version -TimeoutSec 5 | Out-Null }
  catch { Log "ollama not responding; starting ollama serve"; Start-Process $ollama -ArgumentList "serve" -WindowStyle Hidden; Start-Sleep 15 }
}
function Run-Label($label, [string[]]$a) {
  for ($try = 1; $try -le 20; $try++) {
    Ensure-Ollama
    Log "START $label (attempt $try)"
    & $py @a 2>&1 | ForEach-Object { $s = "$_"; if ($s -notmatch '^progress ' -or $s -match '^progress \d*00/') { Log $s } }
    $code = $LASTEXITCODE
    Log "END $label exit=$code"
    if ($code -eq 0) { return $true }
    Start-Sleep 60
  }
  return $false
}
Log "=== run_comments_v2.ps1 started pid=$PID ==="
$posts = "data/raw/comments_v2_posts.csv"
# After a temperature-0 retry, ids that still return no parseable JSON are listed in
# outputs/comments_v2_parse_failures.csv and left out of comments_v2_posts_merge.csv (same as label_posts_merge.csv).
if (Test-Path "data/raw/comments_v2_posts_merge.csv") { $posts = "data/raw/comments_v2_posts_merge.csv" }
$cache = "data/raw/llm_labels_comments_v2.jsonl"
$common = @("llm_label.py","--backend","ollama","--base-url","http://127.0.0.1:11434/v1","--num-ctx","4096","--batch-size","1")
$ok = Run-Label "comments_v2" ($common + @("--posts",$posts,"--out","outputs/llm_labels_comments_v2.csv","--cache",$cache))
if ($ok) {
  Log "START merge-validate comments_v2"
  & $py llm_label.py --merge-validate --expect $posts --caches $cache --out outputs/llm_labels_comments_v2.csv 2>&1 | ForEach-Object { Log "$_" }
  $m1 = $LASTEXITCODE; Log "END merge-validate comments_v2 exit=$m1"
  Log "START merge-validate submissions+comments_v2"
  & $py llm_label.py --merge-validate --expect data/raw/label_posts_merge.csv --expect $posts --caches data/raw/llm_labels_submissions.jsonl --caches $cache --out data/raw/llm_labels_with_comments_v2.csv 2>&1 | ForEach-Object { Log "$_" }
  $m2 = $LASTEXITCODE; Log "END merge-validate combined exit=$m2"
  if ($m1 -eq 0 -and $m2 -eq 0) {
    Log "START run_after_llm comments_v2"
    & $py run_after_llm.py --llm data/raw/llm_labels_with_comments_v2.csv --comment-meta outputs/comments_v2_meta.csv --comments-rebalanced --comment-n-boot 1000 --n-boot 1000 --out outputs/after_llm_comments_v2 2>&1 | ForEach-Object { Log "$_" }
    Log "END run_after_llm exit=$LASTEXITCODE"
  }
} else { Log "labeling did not finish cleanly; rerun run_comments_v2.ps1 to resume" }
Log "=== run_comments_v2.ps1 finished ==="
