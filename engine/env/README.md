# Engine environments

- `.venv-engine` (`engine.lock`): the engine runtime and its tests. Python 3.12, torch 2.13.0+cu130, transformers 5.14.1 and
  flashinfer-python 0.7.0.post1 (the same pins as the locked vllm-omni environment). It also holds minicpmo-utils 1.0.6, installed
  without dependencies, for the `stepaudio2` Token2Wav modules.
- `.venv-hfref` (`hfref.lock`): the HF reference stack from the model card. Python 3.10, transformers 4.51.0, torch 2.8.0+cu128,
  minicpmo-utils[all] 1.0.6. Parity references are produced here (`tools/hfref/`) and compared in the engine environment.

The engine reuses the checkpoint's Apache-2.0 processor and encoder code. One method is adapted for transformers 5.x:
`MiniCPMWhisperEncoderLayer.forward`, because `WhisperAttention` returns two values instead of three
(`engine/models/multimodal.py`).

Reference audio: the HF stack resamples with librosa 0.9 (resampy). The canonical effective inputs, and therefore the engine,
use librosa 1.0 (soxr) at 16 kHz with PCM16 (`results/effective_inputs`). This is the definition recorded in round 0.
