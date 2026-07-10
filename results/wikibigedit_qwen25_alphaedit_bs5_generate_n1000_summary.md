# WikiBigEdit Qwen2.5 AlphaEdit Generation Result

Experiment:
- Dataset: WikiBigEdit
- Model: Qwen2.5-7B-Instruct
- Editor: AlphaEdit
- Sequential edit: true
- Batch size: 5
- Evaluation mode: autoregressive generation, no teacher forcing
- Requested size: 1000
- Completed checkpoint: 1000
- Case range: 0-999
- Raw result: `logs/WikiBigEdit_Qwen2.5-7B-Instruct_AlphaEdit_N=1000_Sequential=True_20260710_090626_wikibigedit_qwen25_alphaedit_bs5_generate_n1000_Checkpoint_1000_results.json`

Summary:

| Metric | Pre | Post |
| --- | ---: | ---: |
| rewrite_acc | 0.0107 | 0.9553 |
| rephrase_acc | 0.0108 | 0.6219 |
| portability.mhop_acc | 0.2221 | 0.4187 |
| portability.personas_acc | 0.2261 | 0.4507 |
| locality.locality_acc | n/a | 0.4159 |

Timing:

| Metric | Value |
| --- | ---: |
| exp_time_so_far_s | 5862.4740 |
| total_edit_time_s | 4189.7101 |
| avg_edit_time_s | 4.1897 |
| total_inference_time_s | 595.1664 |
| avg_inference_time_s | 0.1476 |
| num_of_editor_params | 339476480 |

Validation:
- The raw result JSON parses successfully.
- `individuals` contains 1000 entries.
- The final entry is `case_id: 999`.
- `run.log` contains no `Traceback`, `Exception`, `Killed`, `CUDA out`, `out of memory`, `RuntimeError`, `Segmentation fault`, or `Bus error` markers for this run.
