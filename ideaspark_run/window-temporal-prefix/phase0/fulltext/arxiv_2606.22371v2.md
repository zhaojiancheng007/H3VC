# ZeroGVC: Zero-Shot Generative Video Compression with Autoregressive Diffusion Priors

paper_id: arxiv:2606.22371v2
tier: T3
source_used: html_arxiv
warning: none

## Intro

Generative video compression (GVC) has recently emerged as a promising direction for producing realistic reconstructions at low bitrates, while previous neural video codecs often produce over-smoothed reconstructions
[
14
,
15
,
16
,
12
]
.
Instead of preserving all pixel-level details, GVC methods often leverage powerful generative priors to synthesize visually realistic content from compact representations.
Some approaches introduce adversarial learning into neural video codecs to improve perceptual quality
[
24
,
42
]
. More recently, some works explore more expressive representation spaces, such as discrete VQ-VAE tokens or continuous video latents, for ultra-low bitrate coding
[
28
,
8
]
. Some works exploit diffusion models to recover perceptually pleasing details from highly compact signals
[
19
,
22
,
45
,
21
,
20
,
17
,
41
]
. However, most existing methods require additional training to integrate such priors into video compression systems as reconstruction modules or auxiliary network components, incurring substantial computational cost and limiting their flexibility in adapting to increasingly powerful pretrained generative models.
Training-free generative compression offers an alternative route by directly reusing pretrained generative models without retraining. In the image compression field, several works formulate lossy compression by transmitting compact information associated with diffusion trajectories. DiffC
[
33
]
communicates noisy samples along the diffusion forward process through reverse-channel coding (RCC), and reconstructs images by denoising them with a pretrained diffusion model. Vonderfecht and Liu
[
35
]
further provide a more complete implementation of DiffC by resolving practical bottlenecks in reverse-channel coding. In parallel, DDCM
[
27
]
and Turbo-DDCM
[
34
]
replace continuous Gaussian noise in the reverse diffusion process with selections from reproducible codebooks, and transmit only the compact indices to specify the denoising trajectory. Recent video extensions further demonstrate the potential of training-free GVC. Free-GVC
[
18
]
extends diffusion-based trajectory compression to video latents with temporal coherence based on RCC, while GVCC
[
44
]
converts the deterministic rectified-flow ODE of video diffusion models into an equivalent SDE, enabling DDCM-style codebook selection for zero-shot video compression. Despite this progress, a low-delay coding scheme for zero-shot generative video compression remains underexplored. Existing training-free methods operate at the group-of-pictures (GOP) level with multi-step denoising, which limits both decoding speed and first-frame latency, making them less compatible with low-delay video coding.
Recently, autoregressive (AR) video diffusion models have attracted growing attention for their ability to generate videos in a streaming manner
[
13
,
3
,
43
,
11
,
48
]
. These models combine causal temporal modeling with iterative denoising, rolling out frames sequentially conditioned on previously generated history. This causal, frame-by-frame generation paradigm is naturally aligned with low-delay video coding, as it enables sequential reconstruction using only previously decoded frames. Moreover, recent distillation and training strategies have enabled high-quality few-step AR generation
[
43
,
11
,
48
]
. These observations motivate the question: can such pretrained AR diffusion models be leveraged for low-delay zero-shot generative video compression?
In this paper, we propose ZeroGVC, which encodes the first frame of each GOP using an image codec and reconstructs subsequent P-frames with a pretrained autoregressive diffusion model conditioned on previously decoded frames. The core of ZeroGVC is Codebook-Guided Autoregressive Latent Compression. Beyond the DDPM reverse chain considered by DDCM
[
27
]
and Turbo-DDCM
[
34
]
, we find that the compression scheme of denoising diffusion codebook models is effective in few-step consistency sampling
[
30
]
. Instead of transmitting dense residuals or latent features, ZeroGVC compresses the latent reconstruction trajectory itself. At each denoising step, the pretrained model first predicts a clean latent estimate from the current noisy state. The encoder then measures the target residual between this estimate and the target P-frame latent, and selects a sparse combination of atoms with quantized coefficients from a deterministic Gaussian codebook as the injected noise for the next step. Since the same codebook is regenerated at the decoder from a shared seed, only the selected atom-combination indices and quantized coefficients need to be transmitted. In addition, when the next I-frame is available, ZeroGVC can optionally use it as a bidirectional reference to improve reconstruction quality.
Our contributions are summarized as follows:
•
We introduce ZeroGVC, a training-free generative video compression framework that leverages pretrained autoregressive diffusion priors for low-delay video reconstruction. Unlike existing training-free generative codecs that operate at the GOP level, ZeroGVC reconstructs each P-frame sequentially from previously decoded frames.
•
We find that the compression scheme of denoising diffusion codebook models is effective in few-step consistency sampling, and propose Codebook-Guided Autoregressive Latent Compression, which steers the autoregressive latent denoising trajectory toward the target P-frame by selecting and transmitting compact atom-combination indices and quantized coefficients. We further design an optional bidirectional reference mode to improve reconstruction quality when the next I-frame context is available.
•
Extensive experiments validate the effectiveness of ZeroGVC across several standard video compression benchmarks, demonstrating strong perceptual reconstruction quality at ultra-low bitrates.

## Method

4.1
Pipeline
Figure 1
:
Overview of the proposed ZeroGVC framework. At the encoder, the first frame of each GOP is compressed by an image codec and mapped into the latent space to initialize the autoregressive context, while each subsequent P-frame is represented by compact atom-combination indices and quantized coefficients selected via Top-
M
M
Noise Atom Selection. At the decoder, the same codebook is regenerated from the shared seed, enabling reproducible latent reconstruction from the received atom-combination indices and quantized coefficients before WanVAE decoding. The optional bidirectional reference mode, shown by the gray dashed lines, uses the next decoded I-frame as auxiliary context to mitigate autoregressive error propagation without additional bitrate overhead.
We propose ZeroGVC, a training-free generative video compression framework that exploits a pretrained autoregressive diffusion prior for low-latency video reconstruction. As illustrated in Fig.
1
, given an input video, we divide it into GOPs and denote the
g
g
-th GOP as
𝐗
g
=
{
𝐱
g
,
0
,
𝐱
g
,
1
,
…
,
𝐱
g
,
L
−
1
}
\mathbf{X}_{g}=\{\mathbf{x}_{g,0},\mathbf{x}_{g,1},\ldots,\mathbf{x}_{g,L-1}\}
, where
𝐱
g
,
0
\mathbf{x}_{g,0}
is the I-frame and
{
𝐱
g
,
l
}
l
=
1
L
−
1
\{\mathbf{x}_{g,l}\}_{l=1}^{L-1}
are P-frames. The I-frame is encoded by a pretrained image codec and decoded as
𝐱
^
g
,
0
\hat{\mathbf{x}}_{g,0}
. The WanVAE encoder then maps input video frames and the reconstructed I-frame into latent space, producing the reconstructed I-frame latent
𝐳
^
g
,
0
\hat{\mathbf{z}}_{g,0}
and target P-frame latents
{
𝐳
g
,
l
}
l
=
1
L
−
1
\{\mathbf{z}_{g,l}\}_{l=1}^{L-1}
. The reconstructed I-frame latent is fed to the autoregressive diffusion model to initialize the KV cache used for subsequent P-frame reconstruction.
For each target P-frame latent
𝐳
g
,
l
\mathbf{z}_{g,l}
, the encoder performs codebook-guided autoregressive latent compression conditioned on the current KV cache. At every diffusion step, the pretrained model first predicts a clean latent estimate from the current noisy latent state. Based on this estimate, a sparse combination of atoms with quantized coefficients is selected from a deterministic Gaussian codebook to steer the next state toward
𝐳
g
,
l
\mathbf{z}_{g,l}
. The codebook is generated from a shared seed
S
S
, so only the selected atom-combination indices and quantized coefficients are written into the bitstream. Finally, we obtain the reconstructed latent
𝐳
^
g
,
l
0
\hat{\mathbf{z}}_{g,l}^{0}
and feed it to the autoregressive model to update the KV cache for the next P-frame.
On the decoder side, the I-frame bitstream is first decoded into the reconstructed image
𝐱
^
g
,
0
\hat{\mathbf{x}}_{g,0}
, which is then mapped by the WanVAE encoder to
𝐳
^
g
,
0
\hat{\mathbf{z}}_{g,0}
to initialize the autoregressive context, consistent with the encoder side. For each P-frame, the decoder regenerates the deterministic codebook from the shared seed
S
S
, reads the transmitted atom-combination indices and quantized coefficients, synthesizes the same selected noise, and follows the corresponding diffusion denoising process to reconstruct
𝐳
^
g
,
l
0
\hat{\mathbf{z}}_{g,l}^{0}
. The reconstructed latent is then fed to the autoregressive model to update the cache for the next P-frame. Finally, WanVAE decodes
{
𝐳
^
g
,
l
0
}
l
=
0
L
−
1
\{\hat{\mathbf{z}}_{g,l}^{0}\}_{l=0}^{L-1}
into the reconstructed GOP
𝐗
^
g
=
{
𝐱
^
g
,
0
,
…
,
𝐱
^
g
,
L
−
1
}
\hat{\mathbf{X}}_{g}=\{\hat{\mathbf{x}}_{g,0},\ldots,\hat{\mathbf{x}}_{g,L-1}\}
.
4.2
Codebook-Guided Autoregressive Latent Compression
P-frame compression is achieved by compressing the latent reconstruction trajectory rather than the latent itself. To keep the reconstruction aligned with the target frame, we introduce codebook-guided latent compression into the inference process of causal video diffusion models
[
48
,
11
]
. Although the sampling process differs from the DDPM reverse chain considered by DDCM
[
27
]
and Turbo-DDCM
[
34
]
, we find that selecting and transmitting the injected noise to minimize the target residual remains effective in few-step consistency sampling (Alg.
1
), extending denoising diffusion codebook models to few-step generative reconstruction.
Instead of drawing a Gaussian noise vector at each reinjection step, ZeroGVC selects the noise from a deterministic codebook so that the stochastic trajectory becomes compressible. For the
l
l
-th P-frame in GOP
g
g
, let
𝐳
g
,
l
\mathbf{z}_{g,l}
be the target clean latent and let
ℋ
g
,
l
\mathcal{H}_{g,l}
denote the current KV cache induced by previously reconstructed latents. Starting from an initial noisy latent
𝐳
^
g
,
l
T
\hat{\mathbf{z}}_{g,l}^{T}
, the reconstruction trajectory follows the noise levels
τ
1
>
τ
2
>
⋯
>
τ
N
−
1
\tau_{1}>\tau_{2}>\cdots>\tau_{N-1}
. At step
n
n
, the current clean estimate is denoted by
𝐳
~
g
,
l
0
|
τ
n
−
1
\tilde{\mathbf{z}}_{g,l}^{0|\tau_{n-1}}
, with the first step using the prediction from
𝐳
^
g
,
l
T
\hat{\mathbf{z}}_{g,l}^{T}
. The encoder measures the target residual as
𝐫
g
,
l
n
=
𝐳
g
,
l
−
𝐳
~
g
,
l
0
|
τ
n
−
1
,
\mathbf{r}_{g,l}^{n}=\mathbf{z}_{g,l}-\tilde{\mathbf{z}}_{g,l}^{0|\tau_{n-1}},
(10)
which instantiates the residual definition in Eq.
5
for the target latent
𝐳
g
,
l
\mathbf{z}_{g,l}
. Given the deterministic Gaussian codebook
𝒞
τ
n
\mathcal{C}_{\tau_{n}}
generated from the shared seed
S
S
, the encoder selects a sparse coefficient vector
𝐬
g
,
l
n
,
∗
\mathbf{s}_{g,l}^{n,*}
following the thresholding rule of Turbo-DDCM
[
34
]
.
The selected noise
𝜼
g
,
l
n
,
∗
\boldsymbol{\eta}_{g,l}^{n,*}
is synthesized following Eq.
8
. The next noisy state is obtained by reinjecting this selected noise into the clean prediction,
𝐳
^
g
,
l
τ
n
=
𝐳
~
g
,
l
0
|
τ
n
−
1
+
τ
n
2
−
ϵ
2
​
𝜼
g
,
l
n
,
∗
.
\hat{\mathbf{z}}_{g,l}^{\tau_{n}}=\tilde{\mathbf{z}}_{g,l}^{0|\tau_{n-1}}+\sqrt{\tau_{n}^{2}-\epsilon^{2}}\boldsymbol{\eta}_{g,l}^{n,*}.
(11)
The resulting clean prediction
f
θ
​
(
𝐳
^
g
,
l
τ
n
,
τ
n
;
ℋ
g
,
l
)
f_{\theta}(\hat{\mathbf{z}}_{g,l}^{\tau_{n}},\tau_{n};\mathcal{H}_{g,l})
is used as the current estimate for the next step.
Only the selected atom-combination index and its associated quantized coefficients are transmitted at each step. Since both the initial noisy latent and
𝒞
τ
n
\mathcal{C}_{\tau_{n}}
are generated from the shared seed, the decoder synthesizes the same
𝜼
g
,
l
n
,
∗
\boldsymbol{\eta}_{g,l}^{n,*}
and performs P-frame latent reconstruction consistently with the encoder.
4.3
Bidirectional Reference Mode
The basic mode reconstructs each P-frame using only past decoded latents. We additionally provide a reference mode, indicated by the gray dashed lines in Fig.
1
, that can use the next decoded I-frame as an auxiliary context when it is available. Specifically, for the
g
g
-th GOP, both encoder and decoder use the next I-frame latent
𝐳
^
g
+
1
,
0
\hat{\mathbf{z}}_{g+1,0}
as an additional attention context when reconstructing P-frames in
𝐗
g
\mathbf{X}_{g}
. This shared future reference serves as a stable long-range anchor that mitigates error propagation and improves reconstruction quality in the autoregressive decoding process.
The reference is implemented through additional KV cache entries. A direct append operation, however, would assign the next I-frame an incorrect temporal position: the reference keys and values would be treated as if they immediately followed the current P-frame, while their actual position is at the beginning of the next GOP. To preserve the relative temporal positions encoded by rotary position embeddings (RoPE), we store the future reference keys before rotation and re-apply RoPE using their true future frame indices. This ensures that attention scores between current queries and future reference keys reflect their correct temporal displacement. The optional bidirectional reference mode changes only the available attention context and introduces no additional bitrate overhead.
