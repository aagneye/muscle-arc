# Local backup submit at UTC midnight if VM waiter fails
$token = (Get-Content "$env:USERPROFILE\.kaggle\access_token" -Raw).Trim()
@{username='aagneye'; key=$token} | ConvertTo-Json -Compress | Set-Content "$env:USERPROFILE\.kaggle\kaggle.json" -Encoding ascii
$env:KAGGLE_API_TOKEN = $token
$comp = "umud-challenge-muscle-architecture-in-ultrasound-data"
$until = [DateTime]::Parse("2026-09-10T00:02:00Z").ToUniversalTime()
while ([DateTime]::UtcNow -lt $until) {
  $left = ($until - [DateTime]::UtcNow).TotalSeconds
  Write-Host "utc=$([DateTime]::UtcNow.ToString('o')) remaining=$([int]$left)"
  Start-Sleep -Seconds ([Math]::Min(300, [Math]::Max(5, $left)))
}
Write-Host "SUBMIT_WINDOW"
kaggle competitions submit -c $comp -f C:\Projects\muscle-arc\submissions\submission_dltrack_official.csv -m "dltrack-official-doCalculations-phaseA" 2>&1 | Tee-Object -FilePath C:\Projects\muscle-arc\experiments\local_midnight_submit.log
Start-Sleep -Seconds 20
kaggle competitions submit -c $comp -f C:\Projects\muscle-arc\submissions\submission_dltrack_win.csv -m "dltrack-win-chrome-letterbox-soft" 2>&1 | Tee-Object -FilePath C:\Projects\muscle-arc\experiments\local_midnight_submit.log -Append
Start-Sleep -Seconds 45
$headers = @{ Authorization = "Bearer $token" }
$subs = Invoke-RestMethod -Headers $headers -Uri "https://api.kaggle.com/v1/competitions/submissions/list/${comp}?pageSize=5"
$subs | ConvertTo-Json -Depth 4 | Set-Content C:\Projects\muscle-arc\experiments\post_midnight_scores.json
foreach ($x in $subs) {
  Write-Host "$($x.date) | $($x.publicScore) | $($x.status) | $($x.description)"
}
