# Sequential bounded follow-up to this task's existing local exact trial.
$ihesCurrent = Get-Process -Id 52380 -ErrorAction SilentlyContinue
if ($ihesCurrent -and $ihesCurrent.ProcessName -eq 'twsearch-pdb') {
    Wait-Process -Id 52380 -ErrorAction SilentlyContinue
}
Set-Location -LiteralPath 'C:/Users/and-l/cayley'
$env:IHES_CORNER_PDB_DIR = 'C:/Users/and-l/cayley/data/ihes_pdb'
$env:IHES_PREFIX_RANK_FILE = 'C:/Users/and-l/cayley/data/ihes_pdb/prefix_ranks.txt'
& 'C:/Users/and-l/cayley/.venv/Scripts/python.exe' -u scripts/24_twsearch_ladder.py `
    --baseline submission_ihes.csv --journal data/ihes_pdb/ordered_L24_trial.jsonl `
    --window-length 24 --full-path-only --pids 106 592 680 706 764 810 936 `
    --exe data/ihes_pdb/twsearch-pdb-ranked.exe --threads 16 --memory-mib 8192 `
    --min-depth 22 --max-depth 22 --time-limit 600 `
    --out submissions/ihes_pdb_ordered_L24_trial.csv
exit $LASTEXITCODE
