# Third-party dependencies

H3VC's MIT license applies to this repository's original code.

`tokenizer/h3/` contains the MiniMax-H3 inference VAE source used by H3VC,
under Apache-2.0; see `tokenizer/h3/LICENSE` and the source file headers.
Source: https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/FL2VA/video_vae .
The VAE implementation is retained; trailing blank lines are trimmed. H3VC supplies the construction
config in `tokenizer/h3/config.py` and loads weights from its Stage3 checkpoint.

PyTorch, torchvision, Diffusers, Transformers, Accelerate, OpenCV, NumPy,
Pillow, einops, safetensors, and huggingface-hub retain their respective licenses.
No datasets or pretrained weights are bundled in this repository.
