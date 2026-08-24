### Run summary ###
summary = {
    'chain_complete': not chain_stopped,
    'stages_remaining': [s for s in CFG['stages_to_run'] if s not in completed_paths],
    'stages_trained': {k: str(v) for k, v in STAGE_OUTPUTS.items()},
    'model_under_eval': str(V_ONLY_PATH),
    'eval_label': EVAL_LABEL,
    'canary': CANARY_RESULTS,
    'bench_path_lengths': BENCH_RESULTS,
    'solve': SOLVE_STATS,
    'cfg': CFG,
}
with open(WORK / 'run_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, default=str)

print('=' * 70)
print('RUN SUMMARY')
print('=' * 70)
if summary['chain_complete']:
    print('  chain: COMPLETE')
else:
    print(f"  chain: PAUSED at wall budget - re-run to continue; "
          f"remaining: {summary['stages_remaining']}")
for k, v in summary['stages_trained'].items():
    print(f'  trained {k}: {v}')
print(f'  evaluated: {EVAL_LABEL}')
for label, res in CANARY_RESULTS.items():
    print(f"  canary [{label}]: V(solved)={res['V_solved']:+.3f} "
          f"V(d1)={res['V_d1']:+.3f} sat_gap={res['sat_gap']:+.1f} "
          f"V@80={res['V_at_d80']:.1f} std@20={res['V_std_d20']:.2f}")
if BENCH_RESULTS:
    print(f'  bench: {BENCH_RESULTS}')
if SOLVE_STATS:
    print(f'  solve: {SOLVE_STATS}')
print('full details in run_summary.json / cfg.json')
