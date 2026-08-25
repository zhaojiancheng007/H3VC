# JoyAI-Video-Edit: Real-Time Open-Ended Video Editing with Autoregressive Diffusion

paper_id: arxiv:2608.03974v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

Instruction-guided video editing is evolving from an offline post-production tool into a continuously available visual capability.
Recent diffusion-based editors support increasingly precise and preference-aligned image manipulation
[
24
,
18
]
, as well as diverse video transformations including object manipulation, appearance modification, background replacement, and stylization
[
23
,
15
,
34
,
9
,
19
]
.
Meanwhile, efficient latent representations and selective token processing have reduced the cost of high-resolution video synthesis and localized video editing
[
25
,
31
]
.
This progress opens up applications in live broadcasting, video communication, interactive entertainment, and real-time content creation, where edited frames must be produced as the source video arrives.
Unlike editing a predefined short clip, however, streaming video editing must jointly provide causal output, low response latency, bounded computation and memory, and stable editing quality over an unknown duration.
Most high-quality video editors are designed for offline, fixed-length inputs.
They process an entire clip using bidirectional or global temporal interactions and emit the result only after iterative denoising has completed
[
36
,
29
,
39
]
.
Although full temporal context benefits short-clip coherence, dependence on future frames prevents causal output.
Moreover, temporal tokens, attention states, and activation memory grow with the input length, while bidirectional computation limits the reuse of cached states
[
33
,
37
]
.
Applying an offline editor independently to consecutive clips is not a satisfactory alternative: it repeatedly processes overlapping context and can introduce visible discontinuities at clip boundaries.
A practical streaming editor must instead reuse a bounded temporal state and maintain nearly constant incremental cost as the stream grows.
Causalizing an offline editor alone does not solve this problem.
During training, an autoregressive model typically conditions on clean target history, whereas at inference time it consumes its own imperfect predictions.
This train–inference mismatch causes reconstruction errors, color deviations, and appearance changes to propagate through the generated history and accumulate into long-term drift
[
14
,
11
]
.
The challenge is particularly acute for video editing: in addition to temporal coherence, every output chunk must remain aligned with the current source chunk, preserve untargeted identity, geometry, motion, and background content, and consistently apply the requested transformation.
Generated history promotes continuity but may propagate errors, whereas the source condition preserves fidelity but may weaken editing consistency if not properly balanced.
Furthermore, the few-step generation required for real-time throughput introduces an additional gap between the iterative, classifier-free-guided training target and the single-branch model used at deployment.
Streaming video editing therefore requires causal adaptation, source-faithful few-step distillation, and explicit optimization against long-horizon error accumulation.
We present
JoyAI-Video-Edit
, a 16B-parameter autoregressive diffusion framework for real-time, open-ended video editing.
The model consists of an MLLM-based condition encoder, a causal video VAE, and a multimodal diffusion transformer, and supports both instruction-guided video-to-video editing and reference-conditioned image-and-video-to-video editing.
Starting from a strong bidirectional editor trained through a progressive image and video generation-and-editing curriculum, we convert the model into a chunk-wise causal editor.
Attention is bidirectional within each chunk and causal across chunks.
A sliding temporal window retains a fixed number of recent chunks together with the first chunk as a global sink, thereby bounding the temporal state and per-chunk computation independently of stream duration.
We first train the causal editor with clean-history teacher forcing and subsequently replace the clean history with detached, model-generated estimates through resampling forcing
[
11
]
.
This adaptation exposes the model to the history distribution encountered during deployment and reduces the discrepancy between training and autoregressive inference.
To enable real-time inference without sacrificing editing quality, we introduce Source-Anchored Distribution Matching Distillation (SA-DMD), which distills the iterative diffusion process into a two-step generator.
SA-DMD guides the real-score teacher independently along the text-conditioning and source-fidelity axes, using each temporally aligned source chunk to counteract drift from imperfect autoregressive history.
The source-aware guidance is applied only to the distillation target and absorbed into the generator, allowing the deployed model to preserve source fidelity with a single conditional branch.
We further introduce Long-Horizon Autoregressive Distillation, which performs segmented optimization over extended rollouts and directly supervises states affected by accumulated autoregressive errors while keeping training memory bounded.
Together, chunk-wise autoregressive adaptation, SA-DMD, and long-horizon optimization form a unified training framework that addresses causal generation, source drift, few-step acceleration, and long-term stability.
To support systematic evaluation under sustained streaming, we also construct LongV2VBench, a long-video editing benchmark covering representative global and local editing tasks.
As illustrated in Figure
1
, JoyAI-Video-Edit supports diverse instruction-guided transformations while preserving subject identity, motion, spatial structure, and regions unrelated to the requested edit.
The edited appearance remains coherent as new source frames continuously arrive, demonstrating the model’s ability to maintain both editing consistency and source fidelity over extended streams.
Extensive automatic and human evaluations show that JoyAI-Video-Edit substantially outperforms existing streaming editors and remains competitive with strong offline systems on both short- and long-video editing.
By applying SA-DMD, we reduce the diffusion denoising process to only two steps while retaining high editing quality and temporal consistency.
Together with bounded-history KV caching, FP8 quantization, and an optimized VAE pipeline, this two-step generator enables the complete system to perform end-to-end
720
​
p
720\mathrm{p}
video editing at approximately 30 FPS on a single Nvidia B200 GPU.

## Method

3.1
Model Architecture
As illustrated in Figure
3
, JoyAI-Video-Edit is a unified autoregressive diffusion model designed for video editing. It consists of a multimodal large language model (MLLM), a causal video variational autoencoder (VAE), and a multimodal diffusion transformer (MM-DiT). Our model supports both V2V and IV2V editing under different input conditions.
MLLM.
Given the first frame of the source video and its corresponding editing instruction, the MLLM jointly processes the visual and textual inputs to extract condition tokens. The first frame provides the appearance and semantic context of the source video, while the instruction specifies the intended transformation. The resulting condition tokens encode both the source-aware visual information and the editing intent, providing semantic guidance to the MM-DiT throughout the denoising process.
Causal VAE.
The causal video VAE encodes the source video into latent sequences and maps the optional reference image into the same latent space. It uses a spatiotemporal compression ratio of
8
×
24
×
24
8\times 24\times 24
, corresponding to a temporal compression factor of
8
8
and spatial compression factors of
24
×
24
24\times 24
. Thus, each latent frame represents eight video frames.
MLLM.
Given the first frame of the source video and its corresponding editing instruction, the MLLM extracts condition tokens that capture the source content and the intended edit.
Figure 3
:
Architecture of JoyAI-Video-Edit.
Our autoregressive diffusion architecture consists of an MLLM, a causal video VAE, and an MM-DiT diffusion backbone.
The MLLM extracts condition tokens from textual and visual inputs, while the VAE projects videos and optional reference images into a shared latent space. The MM-DiT jointly models the condition tokens and latent visual tokens to generate edited-video latents, which are subsequently decoded by the VAE into the final output video.
Figure 4
:
The Training Pipeline of JoyAI-Video-Edit.
LHAD indicates the long-horizon autoregressive distillation.
3.2
Foundation Training
T2V training.
We first conduct progressive T2I pretraining at resolutions of
256
×
256
256\times 256
and
512
×
512
512\times 512
to establish text–visual alignment, scene composition, and high-resolution appearance priors. The resulting T2I checkpoint is then used to initialize T2V training, during which temporal modeling is introduced while the T2I objective is retained to prevent degradation of the learned spatial generation capability.
Training begins with low-resolution videos at 12 and 24 fps and subsequently increases both the spatial resolution and frame rate. This progressive curriculum allows the model to acquire motion dynamics and long-range temporal composition without abruptly disrupting the spatial priors inherited from T2I pretraining. After large-scale pretraining, we perform supervised fine-tuning (SFT) on higher-quality data at the target resolution of
480
​
p
480\mathrm{p}
, followed by continual training (CT) to further consolidate visual quality, motion coherence, and prompt alignment.
Bidirectional V2V training.
Starting from the T2V checkpoint, we first introduce I2I supervision to learn instruction-conditioned transformation and content preservation while retaining the T2V and T2I objectives. The resulting image-editing checkpoint then initializes bidirectional video-editing training. We jointly optimize video-to-video (V2V) and image-and-video-to-video (IV2V) supervision together with the preceding tasks
[
15
]
. V2V extends instruction-conditioned editing over time, whereas IV2V additionally propagates appearance or identity cues from reference images.
Specifically, given a source video
x
s
x^{s}
, an editing instruction represented by the conditional tokens
c
c
, an optional reference image, and the edited target
x
t
x^{t}
, the causal video VAE produces the source, reference, and target latents
z
s
z^{s}
,
z
r
z^{r}
, and
z
0
z_{0}
defined above.
The reference latent
z
r
z^{r}
is omitted for V2V samples without a reference image.
For a sampled noise level
σ
∈
[
0
,
1
]
\sigma\in[0,1]
and Gaussian noise
ϵ
\epsilon
, we construct the noisy target and its flow target as
z
σ
=
(
1
−
σ
)
​
z
0
+
σ
​
ϵ
,
v
⋆
=
ϵ
−
z
0
.
z_{\sigma}=(1-\sigma)z_{0}+\sigma\epsilon,\qquad v^{\star}=\epsilon-z_{0}.
(1)
The bidirectional editor
v
θ
bi
v_{\theta_{\mathrm{bi}}}
is trained with
ℒ
V2V
=
E
⁡
[
‖
v
θ
bi
​
(
z
σ
,
σ
,
c
,
z
s
,
z
r
)
−
v
⋆
‖
2
2
]
.
\mathcal{L}_{\mathrm{V2V}}=\operatorname{E}\left[\left\|v_{\theta_{\mathrm{bi}}}(z_{\sigma},\sigma;c,z^{s},z^{r})-v^{\star}\right\|_{2}^{2}\right].
(2)
Bidirectional target attention exposes the model to the complete edited clip, providing a quality-oriented editing initialization.
Bidirectional target attention establishes source preservation and temporally coherent editing before the streaming constraint is introduced. It therefore serves as the starting point for the causal autoregressive adaptation described next.
Table 1
:
Summary of the multi-stage foundation training curriculum.
Volume is measured by the number of training samples; for T2V stages, it reports the video portion of the training mixture.
Phase
Task
Resolution
fps
LR
Volume
Stage 1: Text-to-Image Pretraining
Pretrain
T2I
256
×
256
256\times 256
–
1
×
10
−
4
1\times 10^{-4}
4.3B
Pretrain
T2I
512
×
512
512\times 512
–
1
×
10
−
4
1\times 10^{-4}
793M
Stage 2: Text-to-Video Training
Pretrain
T2V/T2I
256
​
p
256\mathrm{p}
12/24
1
×
10
−
4
1\times 10^{-4}
370M
Pretrain
T2V/T2I
360
​
p
360\mathrm{p}
24
5
×
10
−
5
5\times 10^{-5}
108M
SFT
T2V/T2I
480
​
p
480\mathrm{p}
24
5
×
10
−
5
5\times 10^{-5}
30M
CT
T2V/T2I
480
​
p
480\mathrm{p}
24
5
×
10
−
5
5\times 10^{-5}
6.5M
Stage 3: Image-Editing Training
SFT
I2I/T2V/T2I
720
​
p
720\mathrm{p}
-
1
×
10
−
4
1\times 10^{-4}
3.2M
Stage 4: Bidirectional Video-Editing Training
SFT
V2V/IV2V/I2I/T2V/T2I
720
​
p
720\mathrm{p}
24
5
×
10
−
5
5\times 10^{-5}
5.3M
CT
V2V/IV2V/I2I/T2V/T2I
720
​
p
720\mathrm{p}
24
5
×
10
−
5
5\times 10^{-5}
1.1M
