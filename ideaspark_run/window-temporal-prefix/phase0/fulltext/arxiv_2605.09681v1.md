# Forcing-KV: Hybrid KV Cache Compression for Efficient Autoregressive Video Diffusion Models

paper_id: arxiv:2605.09681v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

Autoregressive (AR) video diffusion
[
10
,
39
,
44
,
31
,
3
,
48
]
has recently emerged as a compelling paradigm for efficient, streaming text-to-video generation. Unlike conventional bidirectional video diffusion models
[
24
,
17
,
14
,
32
]
that denoise all frames simultaneously, AR video diffusion models produce video chunk by chunk, with each new chunk conditioned on previously generated video content via a key-value (KV) cache.
This paradigm enables long-horizon, variable-length video generation with interactive inputs, while reducing both attention complexity and the latency to the first generated content.
Mainstream approaches build upon the
Self Forcing
[
10
,
7
]
training paradigm, performing self-rollout during training to mitigate error accumulation, as exemplified by the broader family of “forcing” methods
[
10
,
39
,
7
,
20
,
19
,
42
,
41
,
37
,
11
,
16
,
6
]
that have shown strong performance.
However, existing mainstream AR video diffusion models still suffer from substantial attention complexity and severe memory overhead due to the heavy KV cache of historical chunks
[
39
,
20
,
4
]
.
As video generation accumulates over time, the currently generated chunk is forced to attend to increasingly long and redundant visual context, which substantially reduces efficiency especially for long-horizon and high-resolution videos.
For instance, generating a 30-second video at 1080P resolution with Self Forcing
[
10
]
takes over 2 minutes on a single NVIDIA H200 GPU, corresponding to a generation speed of 1.71 FPS considering only the overhead within the diffusion transformer (DiT). Moreover, the KV cache alone consumes more than 60 GB of GPU memory in this setting, which poses a major obstacle to deployment in memory-constrained scenarios.
To achieve real-time inference, studies have explored sparse attention
[
21
,
1
]
and feature caching
[
22
,
28
]
techniques for AR video diffusion models.
Although effective, such methods neither reduce memory overhead nor operate on the KV cache, which is a distinctive structural component of AR video diffusion models. Recently, Dummy Forcing
[
9
]
observes that certain heads in AR diffusion models concentrate primarily on the currently generated chunk, and accordingly discards the historical context for those heads.
However, it lacks a detailed analysis of the functional heterogeneity across attention heads, and its aggressive compression results in degraded temporal dynamics and discontinuity across chunks (i.e., flickering and broken transitions at chunk boundaries), as shown in
Section
˜
5
.
We posit that for AR video diffusion models, effective context utilization is key to both quality and efficiency. This raises a pivotal question:
Does autoregressive video diffusion model exhibit distinctive patterns in its KV cache utilization?
Our findings suggest an affirmative answer.
We observe markedly distinct attention patterns and functional roles across the attention heads of AR video diffusion models.
Through a series of careful empirical ablation studies in
Section
˜
3
, we categorize the attention heads into the two categories.
Static heads
consistently attend to the current chunk and the most recent frame, which we denote as the
transition anchor frame
, to preserve intra-frame fidelity and visual continuity across autoregressive chunks.
Dynamic heads
capture inter-frame correspondences across the same spatial regions, governing subject consistency and motion dynamics.
Moreover, we find that this head division remains stable across different samples and denoising steps, and generalizes broadly across multiple AR video diffusion models.
Based on these observations, we propose
Forcing-KV
, a hybrid KV cache compression method for AR video diffusion models that decouples static structural patterns from dynamic context utilization.
Forcing-KV
first introduces a one-shot, model-level
offline head profiling
procedure (see
Section
˜
4.1
) that identifies static and dynamic heads based on frame-wise attention mass.
Subsequently, we apply a hybrid KV cache compression strategy. For static heads, we adopt
static structural pruning
(see
Section
˜
4.2
) to consistently preserve the transition anchor frame and prune distant frames. For dynamic heads, we employ
dynamic similarity pruning
(see
Section
˜
4.3
), which computes segment-wise similarity between adjacent frames in the KV cache to retain temporally evolving content while pruning redundant and unchanged content.
To summarize, our main contributions are:
(1)
Novel Pattern Discovery
: We uncover a universal head specialization pattern shared by mainstream autoregressive video diffusion models: transitions across autoregressive chunks are mediated by static heads that concentrate on the transition anchor frame, whereas long-horizon consistency and dynamics are sustained by dynamic heads through inter-frame attention.
(2)
Hybrid KV cache Compression
: Building upon this, we propose
Forcing-KV
, a compression strategy that preserves structurally critical content for static heads while applying dynamic similarity pruning for dynamic heads, decoupling static patterns from dynamic context utilization.
(3)
Extensive Experiments
: Evaluations across models, benchmarks, generation lengths, and resolutions show that
Forcing-KV
is both high-fidelity and efficient: While maintaining quality,
Forcing-KV
achieves up to
1.35
×
\times
and
1.50
×
\times
speedups along with
30%
cache memory reduction on LongLive and Self Forcing at 480P resolution, further scaling to
2.82
×
\times
at 1080P.

## Method

In this section, we investigate the underlying principles of KV cache utilization in AR video diffusion models to motivate the compression strategy.
We begin with intuitive observations of attention head patterns in
Section
˜
3.1
, followed by empirical evidence that verifies the functional roles of different heads in
Section
˜
3.2
, and finally investigate their stability and generalizability in
Section
˜
3.3
.
3.1
Attention Head Pattern of Autoregressive Video Diffusion Models
Video diffusion models typically exhibit a spatial-temporal functional specialization
[
33
,
40
]
. In AR video diffusion models, the introduction of KV cache allows this property to manifest over the evolving context of autoregressive generation.. This naturally raises the following question:
Question 1:
How do AR video diffusion models organize attention over spatiotemporal content during chunk-wise generation?
To address this, we employ models including Wan2.1
[
32
]
, SkyReels-V2
[
3
]
, Self Forcing
[
10
]
, and Longlive
[
39
]
to generate videos using VBench
[
12
]
prompts
1
1
1
We provide the detailed attention map visualization (
Figure
8
) in
Appendix
B
.
.
Through comparison across bidirectional, autoregressive, many-step and few-step video diffusion models, we categorize the attention heads into
static head
and
dynamic head
, and summarize the patterns in
Figure
˜
2
:
Observation 1 (Static and Dynamic Head Pattern):
Static heads consistently attend to the current chunk and the most recent frame, preserving intra-frame fidelity and visual continuity across local autoregressive chunks. Dynamic heads capture the inter-frame evolution of corresponding regions to exploit long-range temporal context.
As illustrated in
Figure
˜
2
, the static head primarily attends to local spatial frames. Consequently, its attention map exhibits a chunk-wise pattern, with consistent attention placed on the currently generated chunk. Concurrently, static heads also place particular attention on the most recent frame in the historical cache, which we refer to as the
transition anchor frame
. We regard this as a distinctive characteristic of autoregressive video diffusion models, where transitions across autoregressive chunks are primarily mediated through local, static attention to the transition anchor frame, rather than to the full set of historical frames.
Through this attention pattern, the static head provides a structural scaffold for the video. We regard it as an invariant and static behavior in autoregressive video generation, independent of the prompts and the specific generated content.
In contrast, the dynamic head exhibits a diagonal stripe pattern with a constant interval in the KV cache. This phenomenon is highly interpretable: since both the number of frames per chunk and the number of tokens per frame are fixed, the same spatial region across different frames appears with a fixed stride along the key dimension.
As a result, the dynamic head associates each generated region with information from the corresponding regions in historical frames (motion, object evolution), enabling the model to exploit long-range temporal context.
Because different spatial regions evolve dynamically over the course of the video, we refer to this head pattern as dynamic.
Figure 2:
Attention head patterns in AR video diffusion models. Static heads focus on intra-frame dependencies and transitions across autoregressive chunks, whereas dynamic heads capture the inter-frame evolution of corresponding regions.
3.2
Functional Properties of Static and Dynamic Heads
Having established an intuitive interpretation of the head patterns, we proceed to further examine the functional roles of the two types of heads.
Question 2:
What are the specific functional roles of the two types of heads, and what context in the KV cache is essential for them?
We investigate this by conducting separate ablation studies that progressively mask the context accessible to each head until all historical frames are removed. Videos are generated using LongLive
[
39
]
, a high-performing model for long-horizon video generation.
For evaluation, we adopt 128 prompts from MovieGen
[
25
]
and use VBench-Long
[
13
]
as benchmark.
Since existing metrics do not adequately capture flickering and broken transitions at chunk boundaries, we introduce an optical-flow-based metric, termed
chunk discontinuity
to measure
2
2
2
The detailed formulation and metric effectiveness are provided in
Appendix
A
.
.
It measures abrupt changes through the difference in optical flow between adjacent video frames.
We summarize our empirical finding as:
Observation 2 (Functional Properties):
Static heads are crucial for visual continuity across autoregressive chunks while being insensitive to distant context. Dynamic heads govern subject consistency and motion dynamics, drawing on global context that is informative yet partially redundant.
As shown in
Figure
˜
3
(a-c), as the number of visible historical frames is progressively reduced, both dynamic degree and consistency score gradually decline for dynamic heads, while remaining nearly unchanged for static heads.
By contrast, masking the most recent frame (transition anchor frame) causes a sharp increase in chunk discontinuity for static heads, indicating significantly more abrupt transitions at chunk boundaries. We hypothesize that this effect further leads to degradation in other metrics.
The above experiments also verify that transitions across autoregressive chunks are primarily mediated through attention to the transition anchor frame, rather than the full historical context.
Moreover, we observe that adjacent frames in autoregressive generation exhibit substantial regional similarity (potentially redundant), with generally high KV cache similarity that varies across different frame segments, as shown in
Figure
˜
3
(d).
These insights provide empirical support for our hybrid compression scheme, that static heads are pruned statically while dynamic heads are pruned based on similarity, decoupling static local patterns from dynamic context utilization.
Figure 3:
Left:
(a-c) Gradually masking contextual information for dynamic heads leads to a progressive decline in dynamic degree and consistency, while masking the transition frame for static heads causes a sharp rise in chunk discontinuity, revealing different functional emphases. (d) The cosine similarity of key states of adjacent frames across different autoregressive steps and different frame segments.
Right:
(e) Principal component analysis (PCA) of attention features from a subset of attention heads, evaluated across one hundred prompt samples and four denoising steps. The observed head functioning is highly stable.
3.3
Stability of Head Properties
Furthermore, we conduct a statistical analysis of the above head properties to address the question:
Question 3:
Do the head properties remain stable, or do they exhibit substantial variation?
To provide a comprehensive study, we experiment on LongLive with 100 standard VBench
[
12
]
prompts across all four denoising steps.
For a random subset of heads, we extract the key states of each latent frame in the KV cache and compute frame-wise attention features. Based on the features, we visualize the distribution using principal component analysis (PCA) as shown in
Figure
˜
3
(e).
Observation 3 (Stability of Head Properties):
Head functional specialization remains stable across samples and denoising steps in its attention patterns.
As shown in
Figure
˜
3
(e), the features of each head form tightly clustered distributions across different samples and denoising steps, with average intra-head divergence (0.16) substantially smaller than average inter-head divergence(0.83). This provides a basis for effective head classification.
Discussion (Autoregressive Distinctiveness):
Prior studies on bidirectional models also identify spatial-temporal head patterns
[
33
]
. Our observations differ in three important aspects. First, we uncover a unique dependency on transition anchor frames that is specific to autoregressive generation. Second, our observation is grounded in the KV cache, characterizing how the query chunk attends to previously generated chunks rather than fully bidirectional attention.
Third, our compression scheme is based on temporal similarity in the KV cache rather than relying on a sparse attention pattern.
Figure 4:
Overview of
Forcing-KV
. We perform offline head profiling to classify attention heads into
Static
and
Dynamic
. During inference, static heads are pruned leveraging the structural pattern, while dynamic heads are pruned adaptively based on segment-wise similarity of adjacent frames. For simplicity, we use one frame per chunk as an example.
