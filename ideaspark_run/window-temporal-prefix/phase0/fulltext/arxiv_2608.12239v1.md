# HAMP-LIC: Hessian-Aware Mixed-Precision Post-Training Quantization for Learned Image Compression

paper_id: arxiv:2608.12239v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

Recent advances in learned image compression (LIC) have established it as a leading paradigm in image and video coding, achieving significant improvements in rate–distortion performance over traditional handcrafted codecs.
By jointly optimizing nonlinear transforms and probabilistic entropy models, LIC systems are able to learn compact latent representations that substantially enhance compression efficiency while preserving reconstruction quality.
Despite these advantages, the computational and memory requirements of modern LIC architectures remain prohibitive for deployment on resource-constrained devices, limiting their practical applicability in real-world scenarios.
In addition to high inference cost, LIC models are also sensitive to numerical precision
[
12
]
. In practice, different hardware platforms exhibit inconsistent behavior in floating-point arithmetic, particularly in entropy modeling and probability estimation. Such discrepancies can lead to encoding–decoding mismatches, resulting in degraded reconstruction quality or even instability across platforms. This problem is far from negligible in practice: as shown in our experiments (Section
IV-B4
), a full-precision
Cheng2020
model fails to decode up to 24 out of 24 Kodak images and 98 out of 100 Tecnick images when encoding and decoding are performed on different devices (CPU and GPU). These issues highlight the need for efficient and numerically robust compression strategies that maintain performance under low-precision computation.
To address these challenges, model quantization has emerged as a promising solution for reducing computational complexity and memory footprint. Existing quantization methods are generally categorized into Quantization-Aware Training (QAT)
[
2
]
and
Post-Training Quantization (PTQ)
[
14
,
13
,
25
,
15
]
.
QAT incorporates quantization effects during training to improve robustness but requires full access to training data and substantial retraining cost.
By treating all modules with a fixed precision, these methods ignore the heterogeneous sensitivity of different modules, which often leads to suboptimal rate–distortion performance under aggressive quantization.
Although the recent FMPQ
[
15
]
explores mixed-precision quantization for LIC,
it remains limited to relatively high precision (e.g., 8-bit average) and does not fully investigate ultra-low-bit regimes.
In contrast, PTQ directly converts a pre-trained FP32 model into a low-bit representation using a small calibration set, eliminating training time.
However, most existing PTQ approaches for LIC adopt uniform bit-width assignment across all parameters:
Hong
et al.
[
14
]
restrict fixed-point inference to the decoder with range-adaptive quantization (RAQ),
He
et al.
[
13
]
first apply PTQ to pre-trained LIC networks without retraining,
and RDO-PTQ
[
25
]
further introduces rate–distortion-aware block-wise reconstruction.
From a broader perspective, existing LIC quantization methods still face a fundamental trade-off between efficiency, accuracy, and generality. Uniform quantization strategies, while simple, fail to reflect the highly non-uniform redundancy across network components, leading to inefficient precision allocation. In contrast, mixed-precision approaches developed for general vision tasks are often tailored to classification networks and rely on heuristic sensitivity metrics or expensive search procedures, making them difficult to transfer to LIC settings with rate–distortion optimization and entropy-constrained latent representations. Moreover, LIC models possess unique structural characteristics, such as hyperprior-based entropy modeling and iterative analysis–synthesis transforms, which further complicate sensitivity estimation and bit allocation. These differences highlight the necessity of a dedicated mixed-precision quantization strategy specifically designed for LIC.
To further improve efficiency and robustness, recent studies have shown that second-order information, particularly the Hessian matrix
[
9
]
, provides a principled measure of layer-wise sensitivity in neural networks. This insight enables more informed precision allocation by identifying modules that are more sensitive to quantization perturbations. Motivated by this observation, we propose a mixed-precision PTQ framework for LIC that enables efficient low-bit compression while ensuring numerical consistency across heterogeneous hardware platforms.
In our method, block-wise sensitivity is estimated using Hessian-based second-order information, and bit-width allocation is formulated as a constrained integer optimization problem under a target compression ratio. The optimization further incorporates rate–distortion-aware objectives to jointly minimize quantization error and task-specific distortion. The proposed framework, termed Hessian-Aware Mixed-Precision PTQ for LIC (HAMP-LIC), follows a four-stage pipeline (Fig.
1
): (1) sensitivity estimation, (2) task-aware metric construction, (3) mixed-precision allocation via constrained optimization, and (4) block-wise reconstruction refinement.
This article substantially extends our DCC 2026 paper, MPP-LIC
[
31
]
.
Beyond the four-stage mixed-precision PTQ framework, it introduces task-aware sensitivity combining Hessian-trace information with rate–distortion degradation,
formulates bit-width allocation as a global size-constrained optimization solved via Pareto-frontier search,
and details block-wise reconstruction using sequential scaling optimization and adaptive rounding.
The main contributions of this work are summarized as follows:
1.
We propose HAMP-LIC, a mixed-precision PTQ framework that integrates Hessian-based second-order sensitivity estimation with rate–distortion-aware optimization for LIC models.
2.
We design a task-aware sensitivity refinement that modulates the Hessian-trace metric with the rate–distortion loss degradation of each block, so that bit-width decisions directly reflect the compression objective rather than generic reconstruction error.
3.
We formulate bit allocation as a constrained integer optimization problem and introduce an efficient Pareto-frontier search strategy that reduces complexity from exponential to tractable scale, combined with progressive block-wise refinement.
4.
We demonstrate that HAMP-LIC achieves up to 4.85
×
\times
model compression with negligible BD-rate degradation, while ensuring cross-platform numerical consistency and eliminating floating-point-induced decoding mismatches.
The remainder of this paper is organized as follows. Section
II
reviews related work on neural network compression, quantization for LIC, and mixed-precision quantization. Section
III
details the four steps of the proposed HAMP-LIC framework. Section
IV
presents the experimental settings, comparisons with state-of-the-art PTQ methods, and ablation studies. Finally, Section
V
concludes this paper.

## Method

In this section, we present the proposed HAMP-LIC framework.
We first introduce the preliminaries (Section
III-A
),
including the block-wise quantization formulation and the rate–distortion objective of LIC models,
and then detail the four steps ((Sections
III-B
–
III-E
)) through which HAMP-LIC applies mixed-precision quantization to the pretrained LIC model.
The complete procedure is summarized in Algorithm
1
.
III-A
Preliminaries
Assume that the LIC model can be partitioned into
L
L
blocks denoted by
{
B
1
,
B
2
,
…
,
B
L
}
\{B_{1},B_{2},\ldots,B_{L}\}
, each associated with learnable weight parameters
{
W
1
,
W
2
,
…
,
W
L
}
\{W_{1},W_{2},\ldots,W_{L}\}
. Each block may consist of one or more layers,
and the quantization of each block is performed independently to enable
fine-grained reconstruction.
The
b
b
-bit uniform quantization operation applied to a weight parameter
W
W
is defined as:
W
b
=
Clip
​
(
⌊
W
/
s
⌉
,
n
,
p
)
\displaystyle W_{b}=\text{Clip}\left(\left\lfloor W/s\right\rceil,n,p\right)
(1)
where
s
s
denotes the quantization scale parameter that controls the mapping
from floating-point values to integers, and
n
n
and
p
p
represent the negative
and positive integer clipping thresholds, respectively, which are determined
by the target bit-width
b
b
as
n
=
−
2
b
−
1
n=-2^{b-1}
and
p
=
2
b
−
1
−
1
p=2^{b-1}-1
for
signed quantization. The rounding operation
⌊
⋅
⌉
\lfloor\cdot\rceil
maps each
scaled value to its nearest integer, and the clipping operation
Clip
​
(
⋅
,
⋅
)
\text{Clip}(\cdot,\cdot)
ensures that
all values remain within the representable range. Following previous
studies
[
25
,
26
]
, we adopt per-channel scale quantization,
where each output channel of a weight tensor is assigned an independent scale
parameter
s
s
, allowing for finer adjustment of the quantization range and
reducing the overall quantization error compared to per-tensor quantization.
For the LIC network, the overall training objective can be formulated as a
rate-distortion loss function defined over the training set:
ℒ
⁡
(
θ
)
=
1
N
​
∑
i
=
1
N
J
⁡
(
x
i
,
x
^
i
,
θ
)
\displaystyle\mathcal{L}(\theta)=\frac{1}{N}\sum_{i=1}^{N}J(x_{i},\hat{x}_{i};\theta)
(2)
where
x
∈
X
x\in X
denotes the input image set,
x
^
∈
X
^
\hat{x}\in\hat{X}
denotes the corresponding reconstructed image set produced by the LIC model,
and
N
=
|
X
|
N=|X|
is the total number of training samples. The per-sample
R-D loss
J
⁡
(
x
i
,
x
^
i
,
θ
)
J(x_{i},\hat{x}_{i};\theta)
combines a distortion term measuring
reconstruction fidelity (
e.g.
, MSE or MS-SSIM) and a rate term penalizing
the estimated bitrate of the latent representation, jointly encouraging the model
to achieve a favorable rate-distortion trade-off.
Algorithm 1
HAMP-LIC: Hessian-Aware Mixed-Precision PTQ for LIC
0:
FP32 model
ℳ
=
{
B
l
}
l
=
1
L
\mathcal{M}=\{B_{l}\}_{l=1}^{L}
; calibration set
𝒳
cal
\mathcal{X}_{\mathrm{cal}}
; bit-width set
B
B
; budget
ϵ
\epsilon
.
0:
Mixed-precision quantized model
ℳ
m
​
p
\mathcal{M}_{mp}
.
1:
// 1. Hessian-based sensitivity
2:
for
l
=
1
l=1
to
L
L
do
3:
Estimate
Tr
⁡
(
H
l
)
\operatorname{Tr}(H_{l})
on
𝒳
cal
\mathcal{X}_{\mathrm{cal}}
using Hutchinson’s method;
4:
end
for
5:
// 2. Build task-oriented sensitivity list
6:
Evaluate the 8-bit reference loss
ℒ
int8
\mathcal{L}_{\mathrm{int8}}
on
𝒳
cal
\mathcal{X}_{\mathrm{cal}}
;
7:
for
l
=
1
l=1
to
L
L
do
8:
for
b
∈
B
b\in B
do
9:
Quantize
B
l
B_{l}
to
b
b
bits and evaluate
ℒ
qp
,
l
​
(
b
)
\mathcal{L}_{\mathrm{qp},l}(b)
;
10:
Ω
l
​
(
b
)
←
Tr
⁡
(
H
l
)
​
|
ℒ
qp
,
l
​
(
b
)
−
ℒ
int8
|
|
ℒ
int8
|
\displaystyle\Omega_{l}(b)\leftarrow\operatorname{Tr}(H_{l})\frac{|\mathcal{L}_{\mathrm{qp},l}(b)-\mathcal{L}_{\mathrm{int8}}|}{|\mathcal{L}_{\mathrm{int8}}|}
;
11:
end
for
12:
end
for
13:
// 3. Mixed-precision allocation
14:
Rank blocks by task-aware sensitivity and enumerate monotone contiguous partitions as in Eq. (
7
);
15:
{
b
l
∗
}
l
=
1
L
←
argmin
{
b
l
}
∑
l
=
1
L
Ω
l
(
b
l
)
\displaystyle\{b_{l}^{*}\}_{l=1}^{L}\leftarrow\operatorname*{argmin}_{\{b_{l}\}}\sum_{l=1}^{L}\Omega_{l}(b_{l})
16:
s.t.
b
l
∈
B
,
1
8
​
L
​
∑
l
=
1
L
b
l
≤
ϵ
\displaystyle\text{s.t.}\quad b_{l}\in B,\quad\frac{1}{8L}\sum_{l=1}^{L}b_{l}\leq\epsilon
;
17:
// 4. Block-wise reconstruction
18:
for
l
=
1
l=1
to
L
L
do
19:
Set
𝒘
^
l
=
Q
⁡
(
𝒘
l
,
s
𝒘
l
)
\widehat{\boldsymbol{w}}^{\,l}=Q(\boldsymbol{w}^{l};s_{\boldsymbol{w}}^{l})
and
𝒃
^
l
=
Q
⁡
(
𝒃
l
,
s
𝒃
l
)
\widehat{\boldsymbol{b}}^{\,l}=Q(\boldsymbol{b}^{l};s_{\boldsymbol{b}}^{l})
using
b
l
∗
b_{l}^{*}
bits;
20:
Optimize
(
s
𝒘
l
∗
,
s
𝒃
l
∗
)
(s_{\boldsymbol{w}}^{l*},s_{\boldsymbol{b}}^{l*})
by minimizing
(
J
^
−
J
0
)
2
(\widehat{J}-J_{0})^{2}
;
21:
Optimize
𝑽
l
∗
\boldsymbol{V}^{l*}
and the scale factors by minimizing
λ
t
​
ℒ
task
+
ℒ
l
​
q
\lambda_{t}\mathcal{L}_{\mathrm{task}}+\mathcal{L}_{lq}
;
22:
end
for
23:
return
ℳ
m
​
p
\mathcal{M}_{mp}
with
{
b
l
∗
}
l
=
1
L
\{b_{l}^{*}\}_{l=1}^{L}
,
𝒔
∗
\boldsymbol{s}^{*}
, and
𝑽
∗
\boldsymbol{V}^{*}
.
III-B
Step 1: Acquire Sensitivity List
Our approach begins with a full-precision pre-trained LIC network
ℳ
\mathcal{M}
and constructs
a per-block sensitivity list to guide the subsequent mixed-precision quantization
process. The sensitivity list characterizes how sensitive each block is to
quantization-induced perturbations, enabling the allocation of higher bit-widths
to more sensitive blocks and lower bit-widths to more robust ones.
A straightforward choice for measuring block sensitivity is the first-order
gradient information. However, first-order metrics are often insufficient for
accurately capturing layer-wise sensitivity, as they do not account for the
curvature of the loss landscape
[
9
]
. To address this limitation,
we adopt second-order information as the primary sensitivity metric.
Specifically, we utilize the Hessian trace of the loss function with respect
to the weights of each block, which reflects the local curvature and provides
a more reliable indicator of how much the model output would change in response
to weight perturbations introduced by quantization.
Formally, for block
B
l
B_{l}
with weight parameters
W
l
W_{l}
, the Hessian matrix
is defined as:
H
l
=
∂
2
ℒ
∂
W
l
2
H_{l}=\frac{\partial^{2}\mathcal{L}}{\partial W_{l}^{2}}
(3)
where
l
∈
{
1
,
…
,
L
}
l\in\{1,\ldots,L\}
and
ℒ
\mathcal{L}
is the loss function defined over the calibration set.
The Hessian trace
Tr
​
(
H
l
)
\text{Tr}(H_{l})
serves as the sensitivity score for block
B
l
B_{l}
, with a larger trace value indicating greater sensitivity to quantization
[
9
]
.
Direct computation of the full Hessian matrix is computationally prohibitive
for large-scale neural networks. To ensure computational tractability, we
instead estimate the Hessian trace efficiently via Hessian-vector products
combined with randomized probing techniques, based on Hutchinson’s
method
[
9
,
8
]
.
Once the Hessian trace is computed for all
L
L
blocks, the sensitivity scores
are collected and ranked to form the sensitivity list
𝒮
=
{
(
B
l
,
Tr
​
(
H
l
)
)
}
l
=
1
L
\mathcal{S}=\{(B_{l},\text{Tr}(H_{l}))\}_{l=1}^{L}
,
which serves as the foundation for bit-width allocation in the subsequent step.
III-C
Step 2: Build Task-Oriented Sensitivity List
While the Hessian trace obtained in Step 1 provides a
geometry-aware measure of each block’s sensitivity to weight perturbations,
it does not directly reflect the impact of quantization on the downstream
compression task. To address this, we introduce a task-constrained sensitivity
metric that couples the second-order curvature information with the task-specific
rate-distortion loss, enabling a more accurate assessment of each block’s
contribution to the overall model performance under quantization.
Specifically, for each block
B
l
B_{l}
we define a task-aware sensitivity
Ω
l
\Omega_{l}
as a composite measure that integrates both the potential sensitivity captured
by the Hessian trace and the task-oriented loss degradation induced by
quantization. Formally,
Ω
l
\Omega_{l}
is expressed as:
Ω
l
​
(
b
l
)
=
Tr
​
(
H
l
)
⋅
|
ℒ
qp
​
(
b
l
)
−
ℒ
int8
|
|
ℒ
int8
|
\Omega_{l}(b_{l})=\text{Tr}(H_{l})\cdot\frac{|\mathcal{L}_{\text{qp}}(b_{l})-\mathcal{L}_{\text{int8}}|}{|\mathcal{L}_{\text{int8}}|}
(4)
where
L
L
is the number of quantizable blocks,
b
l
b_{l}
is the weight bit-width assigned to the
l
l
-th block,
Tr
​
(
H
l
)
\text{Tr}(H_{l})
is the Hessian trace of block
B
l
B_{l}
computed in Step 1,
and
|
⋅
|
|\cdot|
denotes the absolute value.
The term
ℒ
qp
​
(
b
l
)
\mathcal{L}_{\text{qp}}(b_{l})
denotes the task loss
evaluated when the weights of block
B
l
B_{l}
are quantized to
b
l
b_{l}
bits, while
ℒ
int8
\mathcal{L}_{\text{int8}}
denotes the task loss of the reference model whose
weights are uniformly quantized to 8 bits. The ratio
|
ℒ
qp
​
(
b
l
)
−
ℒ
int8
|
/
|
ℒ
int8
|
|\mathcal{L}_{\text{qp}}(b_{l})-\mathcal{L}_{\text{int8}}|/|\mathcal{L}_{\text{int8}}|
measures the relative task loss degradation
introduced by quantizing block
B
l
B_{l}
, which serves as a task-aware weighting
factor to modulate the Hessian-based sensitivity. By combining these two terms,
Ω
l
\Omega_{l}
captures both the geometric sensitivity of each block and its
practical impact on the compression objective.
For the
i
i
-th sample of an LIC model with a hyperprior structure, the
rate-distortion (R-D) loss
J
i
=
J
⁡
(
x
i
,
x
^
i
,
θ
)
J_{i}=J(x_{i},\hat{x}_{i};\theta)
is defined as:
J
i
\displaystyle J_{i}
=
λ
⋅
D
i
+
R
𝒀
^
,
i
+
R
𝒁
^
,
i
\displaystyle=\lambda\cdot D_{i}+R_{\hat{\boldsymbol{Y}},i}+R_{\hat{\boldsymbol{Z}},i}
(5)
=
λ
⋅
D
⁡
(
x
i
,
x
^
i
)
\displaystyle=\lambda\cdot D(x_{i},\hat{x}_{i})
+
𝔼
⁡
[
−
log
2
⁡
(
p
𝒀
^
​
(
𝒀
^
i
∣
𝒁
^
i
)
)
]
+
𝔼
⁡
[
−
log
2
⁡
(
p
𝒁
^
​
(
𝒁
^
i
)
)
]
,
\displaystyle+\mathbb{E}\!\left[-\log_{2}\!\left(p_{\hat{\boldsymbol{Y}}}(\hat{\boldsymbol{Y}}_{i}\mid\hat{\boldsymbol{Z}}_{i})\right)\right]+\mathbb{E}\!\left[-\log_{2}\!\left(p_{\hat{\boldsymbol{Z}}}(\hat{\boldsymbol{Z}}_{i})\right)\right],
where
λ
\lambda
is the Lagrange multiplier that controls the trade-off between
rate and distortion,
R
𝒀
^
,
i
R_{\hat{\boldsymbol{Y}},i}
is the estimated bitrate for
the quantized latent representation
𝒀
^
i
\hat{\boldsymbol{Y}}_{i}
, and
R
𝒁
^
,
i
R_{\hat{\boldsymbol{Z}},i}
is the estimated bitrate for the hyperprior latent
representation
𝒁
^
i
\hat{\boldsymbol{Z}}_{i}
.
D
⁡
(
x
i
,
x
^
i
)
D(x_{i},\hat{x}_{i})
denotes the distortion between the original image
x
i
x_{i}
and its
reconstruction
x
^
i
\hat{x}_{i}
, which is typically measured by MSE or
MS-SSIM depending on the target application.
Given the task-aware sensitivities
{
Ω
l
}
\{\Omega_{l}\}
defined in
Eq. (
4
), the quantization problem is formulated as a constrained
optimization that seeks the optimal per-block bit-width assignment
{
b
l
}
l
=
1
L
\{b_{l}\}_{l=1}^{L}
to minimize the accumulated sensitivity-weighted loss
degradation—an efficient proxy for the R-D loss
J
J
—while satisfying the
model compression ratio budget:
argmin
{
b
l
}
∑
l
=
1
L
Ω
l
(
b
l
)
\displaystyle\operatorname*{argmin}_{\{b_{l}\}}\ \sum_{l=1}^{L}\Omega_{l}(b_{l})
(6)
s
.
t
.
1
8
​
L
​
∑
l
=
1
L
b
l
≤
ϵ
\displaystyle\mathrm{s.t.}\ \frac{1}{8L}\sum_{l=1}^{L}b_{l}\leq\epsilon
where
ϵ
\epsilon
is a hyperparameter controlling the
overall compression ratio relative to 8-bit quantization, which is empirically
set to
ϵ
=
0.75
\epsilon=0.75
in our experiments; a 10% tolerance on the average
bit-width is allowed in practice when enumerating candidate allocations. This constraint ensures that the
average bit-width across all blocks does not exceed a prescribed budget,
effectively limiting the model size overhead introduced by mixed-precision
quantization. The detailed optimization procedure for solving this constrained
problem is described in Section
III-D
.
III-D
Step 3: Mixed-Precision Bit-Width Allocation via Pareto Frontier Search
Given the task-aware sensitivities
{
Ω
l
}
\{\Omega_{l}\}
refined in Step
III-C
,
this step solves the constrained optimization problem in Eq. (
6
)
to determine the bit-width assignment for each block. Let
B
B
denote the set of admissible bit-widths of size
k
k
for each block,
e.g.
,
B
=
{
2
,
4
,
8
}
B=\{2,4,8\}
, and the bit-width
b
l
b_{l}
assigned to each block in
Eq. (
6
) must satisfy
b
l
∈
B
b_{l}\in B
. The set
ℬ
\mathcal{B}
subsequently
represents all feasible combinations of bit-widths across the entire network,
where each element corresponds to a complete assignment
{
b
1
,
b
2
,
…
,
b
L
}
\{b_{1},b_{2},\ldots,b_{L}\}
.
When formulated as a combinatorial search problem, the bit allocation space
grows exponentially with the number of blocks, reaching
|
ℬ
|
=
k
L
|\mathcal{B}|=k^{L}
in the worst case. For example, with
B
=
{
2
,
4
,
8
}
B=\{2,4,8\}
and a model with
L
=
50
L=50
quantizable blocks, the search space is
3
50
≈
7.12
×
10
23
3^{50}\approx 7.12\times 10^{23}
,
rendering exhaustive exploration computationally intractable.
To reduce the search space, we adopt a Pareto frontier approach guided by the
sensitivity list. Blocks are first sorted in descending order of their sensitivity
scores, and the search is reformulated as a partition problem over the sorted
list, restricting assignments to monotonically non-increasing order with respect
to sensitivity rank. The reduced search space is then expressed as:
|
ℬ
|
=
∑
j
=
1
k
(
k
j
)
⋅
(
L
−
1
j
−
1
)
|\mathcal{B}|=\sum_{j=1}^{k}\binom{k}{j}\cdot\binom{L-1}{j-1}
(7)
where
j
j
indexes the number of distinct bit-width levels used,
(
k
j
)
\binom{k}{j}
counts the ways to select
j
j
bit-widths from
k
k
candidates, and
(
L
−
1
j
−
1
)
\binom{L-1}{j-1}
counts the ways to partition
L
L
sorted blocks into
j
j
contiguous groups. This formulation reduces the search space from exponential
to polynomial in
L
L
, making the search computationally feasible. For the same
example (
L
=
50
L=50
,
k
=
3
k=3
), Eq. (
7
) yields
|
ℬ
|
=
3
+
147
+
1176
=
1326
|\mathcal{B}|=3+147+1176=1326
, a reduction of roughly 21 orders of
magnitude compared with the exhaustive search space of
3
50
≈
7.12
×
10
23
3^{50}\approx 7.12\times 10^{23}
. In our implementation with
k
=
3
k=3
, we enumerate the three-level
partitions of the sorted list (the
j
=
3
j=3
terms), which dominate this reduced
space. Each candidate allocation is evaluated by the objective in
Eq. (
6
), and the assignment that minimizes the accumulated
sensitivity while satisfying the model-size budget is selected as the final
bit-width configuration.
III-E
Step 4: Task-Constraint Block-Wise Optimization
After acquiring the optimal bit-width allocation for each block in Step 3, to further
improve the quantization performance, we adopt task-constraint block-wise optimization,
which can be divided by optimization target into two parts: scaling optimization (Section
III-E1
) and rounding optimization (Section
III-E2
).
III-E
1
Scaling Optimization
We apply block-wise quantization to determine the optimal scale factors for weights and biases. The scale factors for the
l
l
-th block are obtained by minimizing the discrepancy between the quantized and full-precision rate-distortion loss:
s
𝒘
l
,
s
𝒃
l
=
arg
⁡
min
s
𝒘
l
,
s
𝒃
l
​
(
J
^
​
(
𝒘
^
l
,
𝒃
^
l
)
−
J
0
​
(
𝒘
l
,
𝒃
l
)
)
2
s_{\boldsymbol{w}}^{l},s_{\boldsymbol{b}}^{l}=\underset{s_{\boldsymbol{w}}^{l},s_{\boldsymbol{b}}^{l}}{\arg\min}\left(\widehat{J}\left(\hat{\boldsymbol{w}}^{l},\hat{\boldsymbol{b}}^{l}\right)-J_{0}\left(\boldsymbol{w}^{l},\boldsymbol{b}^{l}\right)\right)^{2}
(8)
where
J
^
​
(
𝒘
^
l
,
𝒃
^
l
)
\widehat{J}(\hat{\boldsymbol{w}}^{l},\hat{\boldsymbol{b}}^{l})
denotes the rate-distortion loss evaluated with the quantized weights
𝒘
^
l
\hat{\boldsymbol{w}}^{l}
and biases
𝒃
^
l
\hat{\boldsymbol{b}}^{l}
of the
l
l
-th block,
J
0
​
(
𝒘
l
,
𝒃
l
)
J_{0}(\boldsymbol{w}^{l},\boldsymbol{b}^{l})
denotes the loss of the full-precision counterpart, and
s
𝒘
l
s_{\boldsymbol{w}}^{l}
,
s
𝒃
l
s_{\boldsymbol{b}}^{l}
are the scale factors for the weights and biases of the
l
l
-th block, respectively. The optimization proceeds sequentially from the first block to the last. When optimizing the
l
l
-th block, all preceding blocks have already been quantized with fixed parameters, while all subsequent blocks retain their full-precision parameters.
III-E
2
Rounding Optimization
Rounding-to-nearest is a common default in PTQ but is suboptimal in general. Following AdaRound
[
24
]
, AdaQuant
[
17
]
, and BRECQ
[
22
]
, we adopt an adaptive rounding strategy in which a learnable variable
𝑽
\boldsymbol{V}
is introduced to parameterize the rounding of weights, formally expressed as:
arg
⁡
min
𝑽
​
‖
Λ
⁡
(
𝒘
l
​
𝒙
l
)
−
Λ
⁡
(
𝒘
^
l
​
𝒙
l
)
‖
2
+
β
​
f
reg
​
(
𝑽
)
\underset{\boldsymbol{V}}{\arg\min}\left\|\Lambda\left(\boldsymbol{w}^{l}\boldsymbol{x}^{l}\right)-\Lambda\left(\widehat{\boldsymbol{w}}^{l}\boldsymbol{x}^{l}\right)\right\|^{2}+\beta f_{\text{reg}}(\boldsymbol{V})
(9)
where
Λ
⁡
(
⋅
)
\Lambda(\cdot)
denotes the activation function and
β
​
f
reg
​
(
𝑽
)
\beta f_{\text{reg}}(\boldsymbol{V})
is a differentiable regularization term, weighted by
β
\beta
, that encourages the rounded weights
𝒘
^
\widehat{\boldsymbol{w}}
to converge to integer values
[
24
]
.
The per-block quantization loss is accordingly defined as:
ℒ
l
​
q
=
‖
Λ
⁡
(
𝒘
l
​
𝒙
l
)
−
Λ
⁡
(
𝒘
^
l
​
𝒙
^
l
)
‖
2
+
β
​
f
reg
​
(
𝑽
)
\mathcal{L}_{lq}=\left\|\Lambda\left(\boldsymbol{w}^{l}\boldsymbol{x}^{l}\right)-\Lambda\left(\widehat{\boldsymbol{w}}^{l}\widehat{\boldsymbol{x}}^{l}\right)\right\|^{2}+\beta f_{\text{reg}}(\boldsymbol{V})
(10)
where
𝒙
^
l
\widehat{\boldsymbol{x}}^{l}
denotes the quantized input to the
l
l
-th block, accounting for the accumulated quantization error from preceding blocks.
The total optimization loss jointly combines the task-level loss and the block-wise quantization loss:
𝒔
∗
,
𝑽
∗
=
arg
⁡
min
𝒔
,
𝑽
​
λ
t
​
ℒ
task
+
ℒ
l
​
q
\boldsymbol{s}^{*},\boldsymbol{V}^{*}=\underset{\boldsymbol{s},\boldsymbol{V}}{\arg\min}\ \lambda_{t}\mathcal{L}_{\text{task}}+\mathcal{L}_{lq}
(11)
where
𝒔
\boldsymbol{s}
denotes the collection of scale factors for both weights and biases across all blocks, and
λ
t
\lambda_{t}
is a balancing hyper-parameter empirically set to
1
1
.
The resulting mixed-precision quantized model
ℳ
m
​
p
\mathcal{M}_{mp}
is obtained with the
optimized scale factors
𝒔
∗
\boldsymbol{s}^{*}
, rounding variables
𝑽
∗
\boldsymbol{V}^{*}
,
and per-block bit-width allocation
{
b
l
}
l
=
1
L
\{b_{l}\}_{l=1}^{L}
.
Fig. 2:
Compression performance evaluation on the Kodak dataset (left) and the CLIC dataset (right). Traditional codecs and unquantized LIC models are represented by solid lines, while quantized LIC models are shown with dashed lines, with quantization bits indicated in brackets [·].
Fig. 3:
Visual comparison of the proposed HAMP-LIC method across different quality levels (Q6 to Q3) on CLIC, Kodak, and Tecnick datasets. PSNR (dB), MS-SSIM, and bit-rate (bpp) are reported for each reconstruction.
