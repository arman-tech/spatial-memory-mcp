$env:PATH = 'C:\Users\jon\AppData\Local\Temp\claude\409aeca82b3b64a0\120806bb-ad38-440a-a1f2-1d2c0b5ad9bb\scratchpad\spikes\rmcp-stdio\bin;' + $env:PATH
$env:SPIKE_HOOK_LOG = 'C:\Users\jon\AppData\Local\Temp\claude\409aeca82b3b64a0\120806bb-ad38-440a-a1f2-1d2c0b5ad9bb\scratchpad\spikes\e50-hook.log'
$payload = '{"session_id":"e50","transcript_path":"x","cwd":"C:\tmp","hook_event_name":"UserPromptSubmit","prompt":"how do we rotate the signing key for the staging cluster"}'
$t = @()
for ($i = 0; $i -lt 52; $i++) {
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = 'spatial-memory-spike'
  $psi.Arguments = 'hook user-prompt-submit --client claude-code'
  $psi.UseShellExecute = $false; $psi.RedirectStandardInput = $true; $psi.RedirectStandardOutput = $true
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $p = [System.Diagnostics.Process]::Start($psi)
  $p.StandardInput.Write($payload); $p.StandardInput.Close()
  $null = $p.StandardOutput.ReadToEnd(); $p.WaitForExit()
  $sw.Stop(); if ($i -ge 2) { $t += $sw.Elapsed.TotalMilliseconds }
}
$s = $t | Sort-Object
'n={0} median={1:N2} ms p95={2:N2} ms max={3:N2} ms min={4:N2} ms' -f $s.Count, (($s[24]+$s[25])/2), $s[47], $s[-1], $s[0]
