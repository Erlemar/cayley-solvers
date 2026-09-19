### Export the value head as a plain distance model (the beam-search checkpoint) ###
V_ONLY_PATH = None
EVAL_LABEL = None


def _existing(p):
    return p is not None and Path(p).exists()


if 'az' in STAGE_OUTPUTS and _existing(STAGE_OUTPUTS['az']):
    az_ckpt = STAGE_OUTPUTS['az']
    V_ONLY_PATH = export_dual_head(az_ckpt, MODELS_DIR / 'az_v_only.pt', head='value')
    export_dual_head(az_ckpt, MODELS_DIR / 'az_pi_only.pt', head='policy')
    EVAL_LABEL = f'this run ({Path(az_ckpt).name})'
else:
    # Evaluate the most advanced stage output whose file is actually present (stale
    # chain-state pointers from older runs may reference files no output carries).
    for _stage in [s for s in STAGE_ORDER if s in STAGE_OUTPUTS][::-1]:
        if _existing(STAGE_OUTPUTS[_stage]):
            V_ONLY_PATH = STAGE_OUTPUTS[_stage]
            EVAL_LABEL = f'this run ({_stage})'
            break

if V_ONLY_PATH is None and REFERENCE_V_ONLY.exists():
    V_ONLY_PATH = REFERENCE_V_ONLY
    EVAL_LABEL = 'shipped reference m_az_v4_v_only'
    print('no freshly trained checkpoint available to evaluate - '
          'using the shipped reference instead')

print('model under evaluation:', V_ONLY_PATH, f'({EVAL_LABEL})')
