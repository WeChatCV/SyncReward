# SyncReward

**[Paper](ARXIV_URL) | [Models](https://huggingface.co/Guan123/SyncReward) | [Data](https://huggingface.co/datasets/Guan123/SyncReward-Data)**

SyncReward scores the audio-visual synchronization of a video clip with a reward in [0, 2].
Stage 1 trains PE-AV audio/video encoders contrastively on real clips; Stage 2 freezes them and
aligns a cross-modal Transformer to human sync ratings of generated videos.

## Code Structure

```
configs/             paper settings (stage1 / stage2 / eval), overridable from the CLI
syncreward/
  data.py            video/audio decoding, 0.5 s segmentation, jsonl datasets
  model.py           PE-AV segment encoder, reward Transformer ([REW; V; MOD; A] -> [0, 2]), checkpoint I/O
  losses.py          Stage 1 contrastive loss; Stage 2 regression + ranking loss
  metrics.py         Spearman, Top-1, pairwise accuracy (paper definitions), bootstrap CIs
  utils.py           config parsing, distributed helpers
train_stage1.py      Stage 1: contrastive pre-training on real clips
train_stage2.py      Stage 2: human alignment with the frozen encoder
evaluate.py          benchmark evaluation (checkpoint or external predictions)
score.py             score your own videos
tools/               convert training checkpoints to the released format
scripts/             torchrun launchers
tests/               CPU smoke test
assets/demo/         before/after post-training videos shown above
```



## Installation

```bash
conda create -n syncreward python=3.12 -y && conda activate syncreward
pip install -r requirements.txt      # versions used for the results below; needs FFmpeg for torchcodec
python -m tests.test_smoke           # CPU check, no weights needed
```

The PE-AV backbone is `[facebook/pe-av-large](https://huggingface.co/facebook/pe-av-large)`; pass
`--encoder_path` for a local copy.

## Models


| Model               | Description                                                     | Weights             |
| ------------------- | --------------------------------------------------------------- | ------------------- |
| `syncreward-stage2` | Stage 2 reward model (the paper's SyncReward), full model, fp32 | [HF](https://huggingface.co/Guan123/SyncReward/tree/main/syncreward-stage2) |
| `syncreward-stage1` | Stage 1 PE-AV encoder (frozen inside `syncreward-stage2`)       | [HF](https://huggingface.co/Guan123/SyncReward/tree/main/syncreward-stage1) |


```bash
hf download Guan123/SyncReward --include "syncreward-stage2/*" --local-dir checkpoints
python score.py --checkpoint checkpoints/syncreward-stage2 a.mp4 b.mp4 --output scores.jsonl
```



## Data

Manifests (`video_path` relative to `--video_root`) with `SHA256SUMS`; see the [data card](https://huggingface.co/datasets/Guan123/SyncReward-Data) for fields.


| File                                      | Target                                   |
| ----------------------------------------- | ---------------------------------------- |
| `stage2_train.jsonl` / `stage2_val.jsonl` | bias-corrected rater consensus in [0, 2] |
| `syncreward_bench.jsonl`                  | mean of up to 3 ratings                  |




## Evaluation

```bash
torchrun --standalone --nproc_per_node=8 evaluate.py --config configs/eval.yaml \
    --checkpoint checkpoints/syncreward-stage2 --data $DATA/syncreward_bench.jsonl --video_root $VIDEOS \
    --output outputs/eval/bench.predictions.jsonl --metrics_output outputs/eval/bench.metrics.json

# any other metric: a jsonl with `id` (or `video_path`) and `reward`
python evaluate.py --predictions other.jsonl --data $DATA/syncreward_bench.jsonl
```

`scripts/eval_bench.sh` wraps the first command. Inference uses a centered 4.75 s window, so
it is deterministic. Result of `syncreward-stage2` from this code on 8 x H20
(about 15 min). All 1,000 per-clip predictions are identical to the paper's:

## Training

```bash
# Stage 1 (paper: 32 GPUs; AudioSet first, then --init_from that run on filtered VGGSound, step 5,500 used)
NNODES=4 NODE_RANK=0 MASTER_ADDR=... DATA_ROOT=... bash scripts/train_stage1.sh

# Stage 2 (paper: 2 nodes x 8 GPUs, 1,275 steps)
NNODES=2 NODE_RANK=0 MASTER_ADDR=... DATA_ROOT=$DATA VIDEO_ROOT=$VIDEOS \
    STAGE1_CKPT=checkpoints/syncreward-stage1 bash scripts/train_stage2.sh
```

Hyperparameters are the defaults in `configs/`. `tools/convert_checkpoint.py` turns a `training_state.pt`
into the released `model.safetensors` + `config.json` format.

The released weights are the paper's original checkpoints, converted without any change. We did not
retrain them for this release; Stage 1 alone takes tens of thousands of steps on 32 GPUs. To
reproduce Stage 2, start from the released `syncreward-stage1`. It is bit-identical to the encoder
inside `syncreward-stage2`. Expect seed-level variation rather than identical weights.

## Demo

Text-to-audio-video generations before and after SyncReward-guided DiffusionNFT post-training, using matched prompts. Each clip is a 5 s mp4 with audio; please play with sound on.

| Case | Before post-training | After post-training |
| :--- | :---: | :---: |
| 1. Spoken dialogue | [case1-before.mp4](assets/demo/case1-before.mp4) | [case1-after.mp4](assets/demo/case1-after.mp4) |
| 2. Singing with a string instrument | [case2-before.mp4](assets/demo/case2-before.mp4) | [case2-after.mp4](assets/demo/case2-after.mp4) |
| 3. Handling a toy gun | [case3-before.mp4](assets/demo/case3-before.mp4) | [case3-after.mp4](assets/demo/case3-after.mp4) |
| 4. Drummer on stage | [case4-before.mp4](assets/demo/case4-before.mp4) | [case4-after.mp4](assets/demo/case4-after.mp4) |
| 5. Drumline | [case5-before.mp4](assets/demo/case5-before.mp4) | [case5-after.mp4](assets/demo/case5-after.mp4) |

Before and after clips are different generations from the same prompt, so frames do not
correspond one-to-one. In case 2 the post-trained output drops the instrument named in the prompt,
illustrating the trade-off between synchronization and prompt fidelity discussed in the paper.

## Citation

```bibtex
coming soon
```



## License

Code: LICENSE_PLACEHOLDER. PE-AV weights follow their original license (Apache-2.0).
