# GEM-4D: Geometry-Enhanced Video World Models for Robot Manipulation

paper_id: arxiv:2605.22882v4
tier: T3
source_used: html_arxiv
warning: none

## Intro

General-purpose robotic manipulation requires policies that can generalize across diverse scenes, objects, tasks, and even embodiments.
Vision-language-action policies
[
28
,
4
]
have shown strong progress by directly mapping observations and language instructions to actions, but such generalization often depends on large-scale robot demonstration data and embodiment-specific training or fine-tuning.
Video world models
[
69
,
20
,
15
,
14
,
74
,
1
,
60
,
75
]
offer a complementary path.
By predicting task-conditioned future observations from the current scene and instruction, they can provide a more general visual representation of how an interaction may unfold, potentially reducing reliance on task- and embodiment-specific action supervision.
However, generic video generation models
[
47
,
64
]
primarily emphasize visual realism, whereas using a video world model for robot planning requires more than photorealistic prediction.
To derive executable actions from predicted futures, a robot must recover precise object and end-effector motions
[
2
]
.
This requires the generated video to preserve reliable inter-frame correspondences, such that pixels corresponding to the same 3D surface point follow physically consistent trajectories across time
[
22
]
.
Current video diffusion models
[
39
,
5
]
, trained largely with pixel- or latent-space reconstruction objectives, provide no explicit guarantee of such consistency
[
58
,
36
]
.
They can produce photorealistic videos in which rigid objects deform non-rigidly, contacts drift, and depth varies inconsistently across frames
[
68
]
—errors that may be visually subtle but can fundamentally break action extraction.
This limitation is structural: reliable inter-frame correspondence depends on camera motion, scene depth, and object motion, whereas standard pixel- or latent-space generation losses do not explicitly constrain these factors.
Existing remedies address this limitation only partially.
Explicit 4D-supervised methods
[
69
,
75
]
add predictions such as RGB, depth, and surface normals, thereby constraining certain geometric quantities. However, they require large-scale annotations and still do not provide a unified correspondence signal that jointly accounts for camera motion, scene depth, and object motion.
Our key observation is that the geometric factors governing inter-frame correspondence are already encoded by modern 4D geometry foundation models.
Models such as PAGE-4D
[
72
]
, Depth Anything V3
[
33
]
, VGGT
[
49
]
and VGGT-omega
[
50
]
estimate dense geometry and camera motion from video, so their intermediate representations capture depth, viewpoint change, and object motion in a unified form.
Rather than supervising correspondence explicitly, we distill these geometry-aware representations into the video backbone
[
65
]
.
This feature-level supervision encourages the generative model to encode the camera, depth, and motion structure needed for consistent inter-frame correspondence, yielding predicted futures that are more reliable for robot action extraction.
We present
GEM-4D
, a world model that operationalizes this principle via feature-level distillation.
During training, a video diffusion transformer is paired with a geometry branch that predicts representations from a pretrained 4D geometry foundation model, conditioned on intermediate video features.
This forces the backbone to internalize correspondence-consistent geometric structure without modifying its output space or adding parameters.
The coupling is asymmetric and efficient: the geometry branch reads from video features but never writes back, and is discarded entirely at inference, yielding a single-stream generator with zero additional cost.
We further introduce an inverse dynamics module that converts correspondence-consistent rollouts into executable 6-DoF end-effector trajectories using off-the-shelf vision foundation models, closing the loop from language instruction to real-world manipulation without task-specific training.
GEM-4D
achieves state-of-the-art performance on both video prediction and geometric consistency across both simulation and realistic scenarios and improves real-world manipulation success from 61% to 81%.
Our contributions are summarized as follows:
1.
Principle.
We formalize the connection between geometry foundation model representations and inter-frame correspondences, showing that geometry supervision acts as a representation-level regularizer that encourages the video backbone to encode correspondence-consistent structure.
2.
Architecture.
We introduce
GEM-4D
, a dual flow-matching framework that distills 4D geometry features into a video backbone via asymmetric conditioning, thereby achieving correspondence-aware generation at zero additional inference cost.
3.
System.
We close the loop from world model to robot control: an inverse dynamics module extracts executable trajectories from generated rollouts, achieving 81% success on real-world Droid tasks (+20 points over the strongest baseline) and 63–82% success on RLBench.

## Method

Figure 2
:
GEM-4D
training.
During training, a video DiT predicts the velocity of the noised video latent, while its intermediate features guide a geometry DiT to predict geometry velocity. This coupled training enforces geometry-consistent generation. During inference, only the video branch is used for efficient generation.
3.1
Problem Formulation
Given an initial observation
𝐈
0
\mathbf{I}_{0}
and a language instruction
c
c
, we learn a world model
M
:
(
𝐈
0
,
c
)
→
{
𝐈
t
}
t
=
1
N
M:(\mathbf{I}_{0},c)\rightarrow\{\mathbf{I}_{t}\}_{t=1}^{N}
that predicts future frames, and design a policy extraction module
P
:
{
𝐈
t
}
t
=
0
N
→
{
𝐚
t
}
t
=
0
N
−
1
P:\{\mathbf{I}_{t}\}_{t=0}^{N}\rightarrow\{\mathbf{a}_{t}\}_{t=0}^{N-1}
that extracts executable actions from the predicted rollout.
For
P
P
to succeed, the generated rollout must preserve
inter-frame correspondences
: pixels depicting the same 3D surface point must evolve consistently across time.
When this property is violated—even if the video appears photorealistic—action extraction becomes unreliable because the underlying 3D structure no longer reflects physically realizable motion
[
2
,
56
]
.
We enforce correspondence consistency during training via
Geometry-Enhanced Velocity Alignment
(Sec.
3.2
) shown in Fig.
2
, and convert correspondence-consistent rollouts into actions via an
Inverse Dynamic System
(Sec.
3.3
).
3.2
Geometry-Enhanced Velocity Alignment
What Governs Inter-Frame Correspondence
We begin by formalizing what determines whether two pixels in adjacent frames correspond to the same physical point.
Let
𝐗
t
∈
ℝ
3
\mathbf{X}_{t}\in\mathbb{R}^{3}
be a scene point observed at pixel
𝐩
t
\mathbf{p}_{t}
in frame
t
t
. Under relative camera motion
(
𝐑
t
→
t
+
1
,
𝐓
t
→
t
+
1
)
(\mathbf{R}_{t\to t+1},\mathbf{T}_{t\to t+1})
and scene flow
Δ
​
𝐗
t
\Delta\mathbf{X}_{t}
, its
projection
𝐩
t
+
1
\mathbf{p}_{t+1}
in frame
t
+
1
t{+}1
is:
𝐩
t
+
1
∼
𝐊
​
[
𝐑
t
→
t
+
1
​
𝐃
⁡
(
𝐩
t
)
​
𝐊
−
1
​
𝐩
t
+
𝐓
t
→
t
+
1
+
Δ
​
𝐗
t
]
.
\mathchoice{\hbox to73.01pt{\vbox to6.89pt{\pgfpicture\makeatletter\hbox{\hskip 36.50592pt\lower-2.44449pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-36.50592pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to73.01pt{\vbox to6.89pt{\pgfpicture\makeatletter\hbox{\hskip 36.50592pt\lower-2.44449pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-36.50592pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to44.96pt{\vbox to4.52pt{\pgfpicture\makeatletter\hbox{\hskip 22.47774pt\lower-1.40833pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-22.47774pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to33.89pt{\vbox to3.26pt{\pgfpicture\makeatletter\hbox{\hskip 16.94434pt\lower-1.04166pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-16.94434pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}\sim\mathchoice{\hbox to30.82pt{\vbox to6.86pt{\pgfpicture\makeatletter\hbox{\hskip 15.411pt\lower 0.0pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-15.411pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to30.82pt{\vbox to6.86pt{\pgfpicture\makeatletter\hbox{\hskip 15.411pt\lower 0.0pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-15.411pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to21.62pt{\vbox to4.8pt{\pgfpicture\makeatletter\hbox{\hskip 10.8094pt\lower 0.0pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-10.8094pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to16.95pt{\vbox to3.43pt{\pgfpicture\makeatletter\hbox{\hskip 8.47485pt\lower 0.0pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-8.47485pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}\Big[\mathchoice{\hbox to107.73pt{\vbox to9.58pt{\pgfpicture\makeatletter\hbox{\hskip 53.86455pt\lower-2.72333pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-53.86455pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to107.73pt{\vbox to9.58pt{\pgfpicture\makeatletter\hbox{\hskip 53.86455pt\lower-2.72333pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-53.86455pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to64.68pt{\vbox to6.39pt{\pgfpicture\makeatletter\hbox{\hskip 32.34189pt\lower-1.58665pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-32.34189pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to48.2pt{\vbox to4.56pt{\pgfpicture\makeatletter\hbox{\hskip 24.10124pt\lower-1.13332pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-24.10124pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}\,\mathchoice{\hbox to93.12pt{\vbox to10pt{\pgfpicture\makeatletter\hbox{\hskip 46.56125pt\lower-2.5pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-46.56125pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to93.12pt{\vbox to10pt{\pgfpicture\makeatletter\hbox{\hskip 46.56125pt\lower-2.5pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-46.56125pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to63.72pt{\vbox to7pt{\pgfpicture\makeatletter\hbox{\hskip 31.86105pt\lower-1.75pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-31.86105pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to50.72pt{\vbox to5pt{\pgfpicture\makeatletter\hbox{\hskip 25.36092pt\lower-1.25pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-25.36092pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}\,\mathchoice{\hbox to57.71pt{\vbox to8.64pt{\pgfpicture\makeatletter\hbox{\hskip 28.85556pt\lower 0.0pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-28.85556pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to57.71pt{\vbox to8.64pt{\pgfpicture\makeatletter\hbox{\hskip 28.85556pt\lower 0.0pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-28.85556pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to37.62pt{\vbox to6.68pt{\pgfpicture\makeatletter\hbox{\hskip 18.80939pt\lower 0.0pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-18.80939pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to28.95pt{\vbox to4.32pt{\pgfpicture\makeatletter\hbox{\hskip 14.47484pt\lower 0.0pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-14.47484pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}\,\mathchoice{\hbox to35.68pt{\vbox to6.39pt{\pgfpicture\makeatletter\hbox{\hskip 17.83907pt\lower-1.94444pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-17.83907pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to35.68pt{\vbox to6.39pt{\pgfpicture\makeatletter\hbox{\hskip 17.83907pt\lower-1.94444pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-17.83907pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to23.49pt{\vbox to4.47pt{\pgfpicture\makeatletter\hbox{\hskip 11.7444pt\lower-1.3611pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-11.7444pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to18.56pt{\vbox to3.19pt{\pgfpicture\makeatletter\hbox{\hskip 9.27766pt\lower-0.97221pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-9.27766pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}\;+\;\mathchoice{\hbox to105.74pt{\vbox to9.58pt{\pgfpicture\makeatletter\hbox{\hskip 52.86732pt\lower-2.72333pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-52.86732pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to105.74pt{\vbox to9.58pt{\pgfpicture\makeatletter\hbox{\hskip 52.86732pt\lower-2.72333pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-52.86732pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to63.28pt{\vbox to6.39pt{\pgfpicture\makeatletter\hbox{\hskip 31.6405pt\lower-1.58665pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-31.6405pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to47.22pt{\vbox to4.56pt{\pgfpicture\makeatletter\hbox{\hskip 23.60818pt\lower-1.13332pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-23.60818pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}\;+\;\mathchoice{\hbox to72.13pt{\vbox to8.36pt{\pgfpicture\makeatletter\hbox{\hskip 36.06691pt\lower-1.5pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-36.06691pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to72.13pt{\vbox to8.36pt{\pgfpicture\makeatletter\hbox{\hskip 36.06691pt\lower-1.5pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-36.06691pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to49.06pt{\vbox to5.8pt{\pgfpicture\makeatletter\hbox{\hskip 24.52943pt\lower-1.0pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-24.52943pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to38.66pt{\vbox to4.18pt{\pgfpicture\makeatletter\hbox{\hskip 19.33049pt\lower-0.75pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-19.33049pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}\Big].
(1)
Pixel in next frame
Intrinsic matrix
Inverse intrinsic matrix
Pixel in current frame
Camera translation
Camera rotation
Depth at
𝐩
t
\mathbf{p}_{t}
Scene flow / object motion
This leads to two immediate insights.
First, pixel-level reconstruction losses cannot enforce correspondences: the mapping from scene geometry to pixel values is many-to-one, so different configurations of
(
𝐃
,
𝐑
,
𝐓
,
Δ
​
𝐗
)
(\mathbf{D},\mathbf{R},\mathbf{T},\Delta\mathbf{X})
can produce visually indistinguishable frames.
A pixel loss can reach zero while the underlying correspondences are entirely wrong.
Second, a model whose internal representations correctly encode
(
𝐃
,
𝐑
,
𝐓
,
Δ
​
𝐗
)
(\mathbf{D},\mathbf{R},\mathbf{T},\Delta\mathbf{X})
necessarily
produces correct
correspondences, since Eq.
1
leaves no remaining degree of freedom.
Geometry foundation models as correspondence encoders.
Models such as PAGE-4D
[
72
]
, Depth Anything V3
[
33
]
,VGGT
[
49
]
,and DUSt3R-family models
[
52
,
67
,
51
]
employ cross-frame transformers supervised on two tasks: dense depth estimation and camera pose regression.
These are precisely the dominant factors in
Eq.
1
: given per-frame depth
𝐃
\mathbf{D}
and relative camera pose
(
𝐑
,
𝐓
)
(\mathbf{R},\mathbf{T})
, the correspondence of any static scene point is fully determined.
For dynamic points, the model must additionally capture how depth changes across frames—which implicitly encodes
object motion
Δ
​
𝐗
\Delta\mathbf{X}
, since a point whose depth evolves inconsistently with the estimated camera motion must be moving independently.
The learned feature representations of these models
therefore encode the complete correspondence structure of
Eq.
1
, despite being supervised only on depth and pose.
Supervising the video backbone to predict these representations implicitly encourages correspondence consistency, without requiring an explicit correspondence loss.
GEM-4D shares REPA’s
[
65
]
philosophy of leveraging foundation-model supervision to improve generative representations. Unlike REPA, which explicitly aligns the diffusion representation with foundation-model features, GEM-4D uses geometry foundation models as auxiliary supervision to encourage a shared video representation that jointly supports RGB denoising and 3D geometric reasoning.
Flow Matching for Latent Video Generation
We adopt flow matching
[
34
]
for latent video generation.
Let
𝐳
0
\mathbf{z}_{0}
be the VAE-encoded video latent
[
29
]
and
𝐳
1
∼
𝒩
⁡
(
𝟎
,
𝐈
)
\mathbf{z}_{1}\sim\mathcal{N}(\mathbf{0},\mathbf{I})
be noise.
A velocity field
𝐯
θ
vid
\mathbf{v}_{\theta}^{\text{vid}}
transports
𝐳
1
\mathbf{z}_{1}
to
𝐳
0
\mathbf{z}_{0}
via the ODE
d
​
𝐳
t
/
d
​
t
=
𝐯
θ
vid
​
(
𝐳
t
,
t
,
c
)
d\mathbf{z}_{t}/dt=\mathbf{v}_{\theta}^{\text{vid}}(\mathbf{z}_{t},t,c)
, trained by regressing against an analytically derived target velocity
𝐯
∗
\mathbf{v}^{*}
:
ℒ
FM
vid
=
𝔼
𝐳
0
,
𝐳
1
,
t
​
[
‖
𝐯
θ
vid
​
(
𝐳
t
,
t
,
c
)
−
𝐯
∗
​
(
𝐳
t
,
t
)
‖
2
2
]
.
\mathcal{L}_{\mathrm{FM}}^{\text{vid}}=\mathbb{E}_{\mathbf{z}_{0},\mathbf{z}_{1},t}\!\left[\,\|\mathbf{v}_{\theta}^{\text{vid}}(\mathbf{z}_{t},t,c)-\mathbf{v}^{*}(\mathbf{z}_{t},t)\|_{2}^{2}\,\right].
(2)
We parameterize the velocity through a Video DiT
[
39
]
with backbone
E
θ
vid
E_{\theta}^{\text{vid}}
and output head
U
θ
vid
U_{\theta}^{\text{vid}}
:
𝐦
t
=
E
θ
vid
​
(
𝐳
t
,
t
,
c
)
,
𝐯
θ
vid
=
U
θ
vid
​
(
𝐦
t
)
,
\mathbf{m}_{t}=E_{\theta}^{\text{vid}}(\mathbf{z}_{t},t,c),\qquad\mathbf{v}_{\theta}^{\text{vid}}=U_{\theta}^{\text{vid}}(\mathbf{m}_{t}),
(3)
where
𝐦
t
\mathbf{m}_{t}
denotes the intermediate features extracted at a mid-level layer.
Under standard flow matching,
𝐦
t
\mathbf{m}_{t}
is optimized only for appearance transport.
The next section describes how we shape
𝐦
t
\mathbf{m}_{t}
to encode correspondence structure.
Correspondence Distillation via Geometry Flow
Given a video sequence
{
𝐈
t
}
t
=
0
T
\{\mathbf{I}_{t}\}_{t=0}^{T}
, a frozen geometry model
G
G
extracts a dense geometric representation:
𝐠
0
=
G
⁡
(
{
𝐈
t
}
t
=
0
T
)
∈
ℝ
T
×
H
P
×
W
P
×
C
.
\mathbf{g}_{0}=G\!\left(\{\mathbf{I}_{t}\}_{t=0}^{T}\right)\in\mathbb{R}^{T\times\frac{H}{P}\times\frac{W}{P}\times C}.
(4)
As established in Sec.
3.2
,
𝐠
0
\mathbf{g}_{0}
encodes the factors
(
𝐃
,
𝐑
,
𝐓
,
Δ
​
𝐗
)
(\mathbf{D},\mathbf{R},\mathbf{T},\Delta\mathbf{X})
in Eq.
1
—it is a dense correspondence representation of the input video.
G
G
remains frozen throughout; it serves as a correspondence teacher whose knowledge we distill into the video backbone.
To perform this distillation, we introduce a parallel flow-matching process over the geometry representation space.
A Geometry DiT
𝐯
ψ
geo
\mathbf{v}_{\psi}^{\text{geo}}
,
conditioned on the video backbone’s intermediate features
𝐦
t
\mathbf{m}_{t}
, predicts the velocity field of the geometry latent:
ℒ
FM
geo
=
𝔼
𝐠
0
,
𝐠
1
,
t
​
[
‖
ℳ
⁡
(
𝐯
ψ
geo
​
(
𝐠
t
,
t
,
𝐦
t
)
)
−
𝐯
∗
​
(
𝐠
t
,
t
)
‖
2
2
]
,
\mathcal{L}_{\mathrm{FM}}^{\text{geo}}=\mathbb{E}_{\mathbf{g}_{0},\mathbf{g}_{1},t}\!\left[\,\|\mathcal{M}(\mathbf{v}_{\psi}^{\text{geo}}(\mathbf{g}_{t},t,\mathbf{m}_{t}))-\mathbf{v}^{*}(\mathbf{g}_{t},t)\|_{2}^{2}\,\right],
(5)
where
𝐠
t
\mathbf{g}_{t}
interpolates between
𝐠
0
\mathbf{g}_{0}
and noise
𝐠
1
∼
𝒩
⁡
(
𝟎
,
𝐈
)
\mathbf{g}_{1}\sim\mathcal{N}(\mathbf{0},\mathbf{I})
;
ℳ
\mathcal{M}
denotes mask mechanism applied to noised geometry feature.
The critical design choice is that the Geometry DiT receives
𝐦
t
\mathbf{m}_{t}
as its
only
scene-level conditioning signal.
It has no direct access to pixels, camera parameters, or depth maps—all scene information must arrive through
𝐦
t
\mathbf{m}_{t}
.
Minimizing
ℒ
FM
geo
\mathcal{L}_{\mathrm{FM}}^{\text{geo}}
therefore requires
𝐦
t
\mathbf{m}_{t}
to contain sufficient information about the geometric factors
(
𝐃
,
𝐑
,
𝐓
,
Δ
​
𝐗
)
(\mathbf{D},\mathbf{R},\mathbf{T},\Delta\mathbf{X})
to predict how the geometry representation evolves over time.
Since these factors determine correspondences (Eq.
1
), the geometry loss could serve as a correspondence loss imposed on the video backbone’s internal representations.
Joint Objective and Geometric Grounding
The training objective is:
ℒ
=
ℒ
FM
vid
+
α
​
ℒ
FM
geo
,
\mathcal{L}=\mathcal{L}_{\mathrm{FM}}^{\text{vid}}+\alpha\,\mathcal{L}_{\mathrm{FM}}^{\text{geo}},
(6)
where
α
\mathbf{\alpha}
balances the two terms. Because both losses share the intermediate representation
𝐦
t
\mathbf{m}_{t}
, the gradient with respect to the video backbone parameters
θ
\theta
decomposes as:
∇
θ
ℒ
=
∇
θ
ℒ
FM
vid
+
α
⋅
∂
ℒ
FM
geo
∂
𝐦
t
⋅
∂
𝐦
t
∂
θ
\mathchoice{\hbox to67.1pt{\vbox to8.58pt{\pgfpicture\makeatletter\hbox{\hskip 33.54874pt\lower-1.74997pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-33.54874pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to67.1pt{\vbox to8.58pt{\pgfpicture\makeatletter\hbox{\hskip 33.54874pt\lower-1.74997pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-33.54874pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to45.39pt{\vbox to6.01pt{\pgfpicture\makeatletter\hbox{\hskip 22.69334pt\lower-1.22499pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-22.69334pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to36.22pt{\vbox to4.29pt{\pgfpicture\makeatletter\hbox{\hskip 18.10846pt\lower-0.87498pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-18.10846pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}=\mathchoice{\hbox to114.35pt{\vbox to10.74pt{\pgfpicture\makeatletter\hbox{\hskip 57.17397pt\lower-1.74997pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-57.17397pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to114.35pt{\vbox to10.74pt{\pgfpicture\makeatletter\hbox{\hskip 57.17397pt\lower-1.74997pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-57.17397pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to73.75pt{\vbox to8.15pt{\pgfpicture\makeatletter\hbox{\hskip 36.8767pt\lower-1.22499pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-36.8767pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to57.05pt{\vbox to5.37pt{\pgfpicture\makeatletter\hbox{\hskip 28.52513pt\lower-0.87498pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-28.52513pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}\;+\;\mathchoice{\hbox to22.56pt{\vbox to4.31pt{\pgfpicture\makeatletter\hbox{\hskip 11.2797pt\lower 0.0pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-11.2797pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to22.56pt{\vbox to4.31pt{\pgfpicture\makeatletter\hbox{\hskip 11.2797pt\lower 0.0pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-11.2797pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to15.77pt{\vbox to3.01pt{\pgfpicture\makeatletter\hbox{\hskip 7.8855pt\lower 0.0pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-7.8855pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to12.97pt{\vbox to2.15pt{\pgfpicture\makeatletter\hbox{\hskip 6.48569pt\lower 0.0pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-6.48569pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}\cdot\mathchoice{\hbox to244.42pt{\vbox to8.62pt{\pgfpicture\makeatletter\hbox{\hskip 122.2096pt\lower-1.67657pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-122.2096pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to244.42pt{\vbox to8.62pt{\pgfpicture\makeatletter\hbox{\hskip 122.2096pt\lower-1.67657pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-122.2096pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to150.34pt{\vbox to6.31pt{\pgfpicture\makeatletter\hbox{\hskip 75.17053pt\lower-1.18611pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-75.17053pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}{\hbox to116.04pt{\vbox to4.32pt{\pgfpicture\makeatletter\hbox{\hskip 58.0186pt\lower-0.84721pt\hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} \lxSVG@begingroup@{stroke} \lxSVG@begingroup@{fill} \lxSVG@setlinewidth{\the\pgflinewidth}\lxSVG@begingroup@{stroke-width} \lx@inpgf@ignorespaces\nullfont\lxSVG@begingroup@{_scopebegin} \lxSVG@closescope \hbox to0.0pt{\lxSVG@begingroup@{_scopebegin} {}{
{{}}\lx@inpgf@ignorespaces\hbox{\hbox{{\lxSVG@begingroup@{_scopebegin} {{}{}{{
{}{}}}{
{}{}}
{{}{{\lx@inpgf@ignorespaces}}}{{}{\lx@inpgf@ignorespaces}}{}{{}{\lx@inpgf@ignorespaces}}
{\lx@inpgf@ignorespaces
}{{{{\lx@inpgf@ignorespaces}}\lxSVG@begingroup@{_scopebegin} \lxSVG@transformcm{1.0}{0.0}{0.0}{1.0}{-58.0186pt}{0.0pt}\lxSVG@begingroup@{transform} \pgfsys@hbox{58}\lxSVG@closescope }}}
\lxSVG@closescope }}}
}
\lxSVG@closescope \hbox to0.0pt{}{{
{}{}{}}}{\lx@inpgf@ignorespaces}{\lx@inpgf@ignorespaces}\hss}\lxSVG@discardpath\lxSVG@closescope \hss}}\lxSVG@closescope\endpgfpicture}}}
(7)
Total gradient
Appearance supervision
Geometry-induced gradient
Balancing weight
Because both losses share the intermediate representation
𝐦
t
\mathbf{m}_{t}
, the gradient decomposes into an appearance term and a geometry-induced term.
The latter propagates correspondence structure into the video backbone through
𝐦
t
\mathbf{m}_{t}
.
The first term drives
𝐦
t
\mathbf{m}_{t}
to encode how pixels move over
time—appearance transport. The second, which is non-zero by
construction since
ℒ
FM
geo
\mathcal{L}_{\mathrm{FM}}^{\text{geo}}
depends on
𝐦
t
\mathbf{m}_{t}
(Eq.
3
), drives
𝐦
t
\mathbf{m}_{t}
to additionally encode
why
they move that way: the depth, camera pose, and scene flow
that determine correspondences (Eq.
1
). At
convergence,
𝐦
t
\mathbf{m}_{t}
must satisfy both objectives, eliminating
representations that explain appearance but violate geometric
consistency.
3.3
Adaptive Inverse Dynamic System
Given generated frames
{
𝐈
t
}
t
=
0
N
\{\mathbf{I}_{t}\}_{t=0}^{N}
, we make use of
Adaptive Inverse Dynamic System (AIDS)
to convert it into executable 6-DoF action trajectories
{
𝐚
t
}
t
=
0
N
−
1
\{\mathbf{a}_{t}\}_{t=0}^{N-1}
, as illustrated in
Fig.
3
.
AIDS introduces two mechanisms that together make action extraction robust to these artifacts in generated results without any task-specific training: (i) a
dual-criterion confidence-gated tracker
that separates gradual drift from catastrophic collapse and invokes heavyweight VLM re-grounding only when warranted, and (ii) a
geometry–kinematics pose fallback
that decouples translation (well-observed from depth) from rotation (temporally smoothed in
S
​
E
​
(
3
)
SE(3)
) whenever learned pose estimation becomes unreliable.
Figure 3
:
Adaptive Inverse Dynamic System.
Given a generated video as input, this system extracts a robot policy through the four steps illustrated in the figure.
3D scene grounding.
We first localize the target object and end-effector (EE) in 3D.
Given the instruction, the depth map, and camera intrinsics (estimated by the geometry foundation model), Qwen3.5-VL
[
57
]
and SAM 2
[
42
,
30
]
generate segmentation masks for the target object and the end effector (EE), which are then used to extract their corresponding point clouds.
We then align the EE CAD to the EE point cloud with FoundationPose
[
55
]
to recover the initial EE translation and rotation
(
𝐑
ee
0
,
𝐓
ee
0
)
∈
S
​
E
​
(
3
)
\left(\mathbf{R}_{\text{ee}}^{0},\mathbf{T}_{\text{ee}}^{0}\right)\in SE(3)
.
Dual-Criterion Confidence-Gated Tracker.
With initial EE pose, we propagate dense keypoints sampled from the mask of end effector
ℳ
ee
\mathcal{M}_{\text{ee}}
through the rollout using CoTracker3
[
25
,
26
]
. Let
𝒱
t
0
⊆
ℳ
ee
\mathcal{V}_{t_{0}}\subseteq\mathcal{M}_{\text{ee}}
be the anchor keypoint set at initial frame, and
𝒱
t
⊆
𝒱
t
0
\mathcal{V}_{t}\subseteq\mathcal{V}_{t_{0}}
the subset still reliably tracked at frame
t
t
. We monitor following two metrics:
s
t
=
|
𝒱
t
|
|
𝒱
t
0
|
∈
[
0
,
1
]
,
Δ
​
s
t
=
s
t
−
s
t
−
1
,
s_{t}=\frac{|\mathcal{V}_{t}|}{|\mathcal{V}_{t_{0}}|}\in[0,1],\qquad\Delta s_{t}=s_{t}-s_{t-1},
(8)
where
s
t
s_{t}
denotes the anchor retention ratio (
s
t
=
1
s_{t}=1
means all anchors remain reliably tracked,
s
t
→
0
s_{t}{\to}0
means tracker failure) and
Δ
​
s
t
\Delta s_{t}
denotes the frame-to-frame change (negative value signals a loss of correspondence).
These two signals separate two qualitatively different failure modes. A
gradual drift
manifests as
s
t
s_{t}
decaying smoothly below the retention threshold
τ
∈
(
0
,
1
)
\tau\in(0,1)
; an
abrupt collapse
, typically caused by frame-level generative artifacts, manifests as a sharp drop
Δ
​
s
t
<
−
δ
\Delta s_{t}<-\delta
, where
δ
>
0
\delta>0
is the drop threshold.
We handle these two failure modes with different interventions:
ℳ
^
ee
t
=
{
re-anchor tracker at
​
t
,
if
​
s
t
<
τ
,
Qwen3
​
.5
​
-
​
VL
​
(
I
t
,
c
)
,
if
​
Δ
​
s
t
<
−
δ
,
ℳ
ee
t
,
otherwise.
\hat{\mathcal{M}}_{\text{ee}}^{\,t}=\begin{cases}\text{re-anchor tracker at }t,&\text{if }s_{t}<\tau,\\[2.0pt]
\mathrm{Qwen3.5\text{-}VL}(I_{t},\,c),&\text{if }\Delta s_{t}<-\delta,\\[2.0pt]
\mathcal{M}_{\text{ee}}^{\,t},&\text{otherwise.}\end{cases}
(9)
The first branch corresponds to a
gradual drift
(resampling fresh keypoints from the most recent reliable mask), and the second to an
abrupt collapse
(re-grounding the EE mask itself via vision-language semantics).
Figure 4
:
Generated Frames to Arm Action.
Starting from an initial observation,
GEM-4D
predicts future frames, which are then converted into executable UF arm actions. The generated frames are displayed with higher brightness to distinguish them from the real frames.
Geometry–kinematics pose fallback.
Given the mask of end effector
ℳ
ee
t
\mathcal{M}_{\text{ee}}^{\,t}
, the frame
𝐈
t
\mathbf{I}_{t}
, the depth
𝐃
t
\mathbf{D}_{t}
, and the EE CAD model, FoundationPose
[
55
]
predicts the pose of EE
(
𝐑
ee
t
,
𝐓
ee
t
)
\left(\mathbf{R}_{\text{ee}}^{t},\mathbf{T}_{\text{ee}}^{t}\right)
together with a confidence
κ
t
∈
[
0
,
1
]
\kappa_{t}\in[0,1]
for each frame.
(
𝐑
ee
t
,
𝐓
ee
t
,
κ
t
)
=
FoundationPose
⁡
(
𝐈
t
,
𝐃
t
,
ℳ
ee
t
,
CAD
)
,
(\mathbf{R}_{\text{ee}}^{t},\mathbf{T}_{\text{ee}}^{t},\kappa_{t})=\mathrm{FoundationPose}\bigl(\mathbf{I}_{t},\mathbf{D}_{t},\mathcal{M}_{\mathrm{ee}}^{\,t},\mathrm{CAD}\bigr),
(10)
When
κ
t
<
κ
∗
\kappa_{t}<\kappa^{\ast}
, where
κ
∗
∈
[
0
,
1
]
\kappa^{\ast}\in[0,1]
is a FoundationPose acceptance confidence threshold, we apply a temporal consistency check and reject the FoundationPose estimate if either the translation jump or the rotation jump exceeds its corresponding threshold:
‖
𝐓
ee
t
−
𝐓
ee
t
−
1
‖
2
>
ϵ
t
,
d
geo
​
(
𝐑
ee
t
,
𝐑
ee
t
−
1
)
>
ϵ
R
.
\|\mathbf{T}_{\mathrm{ee}}^{t}-\mathbf{T}_{\mathrm{ee}}^{t-1}\|_{2}>\epsilon_{t},\quad d_{\mathrm{geo}}(\mathbf{R}_{\mathrm{ee}}^{t},\mathbf{R}_{\mathrm{ee}}^{t-1})>\epsilon_{R}.
(11)
Where
d
geo
​
(
⋅
,
⋅
)
d_{\mathrm{geo}}(\cdot,\cdot)
denotes the geodesic distance on
S
​
O
​
(
3
)
SO(3)
. For rejected frames, we recover the EE translation by back-projecting valid depth pixels within the EE mask and taking their 3D centroid, while the missing rotation is estimated by spherical linear interpolation
[
43
]
between the nearest accepted poses in time. Finally, we’ll have the recovered EE trajectory
{
(
𝐑
ee
t
,
𝐓
ee
t
)
}
t
=
0
N
\{(\mathbf{R}_{\mathrm{ee}}^{\,t},\mathbf{T}_{\mathrm{ee}}^{\,t})\}_{t=0}^{N}
.
Grasp insertion and action synthesis.
From the recovered EE trajectory, we first select the pose closest to the target object as a reference pose, denoted by
(
𝐑
ref
,
𝐓
ref
)
(\mathbf{R}_{\mathrm{ref}},\mathbf{T}_{\mathrm{ref}})
. GraspGen
[
38
]
then predicts a set of grasp candidates
{
(
𝐑
grasp
i
,
𝐓
grasp
i
)
}
i
=
0
M
\{(\mathbf{R}_{\mathrm{grasp}}^{i},\mathbf{T}_{\mathrm{grasp}}^{i})\}_{i=0}^{M}
on the target object point cloud. We rank these candidates by their weighted pose deviation from reference pose, combining translation and rotation errors, and select the best-scoring candidate as the final grasp pose:
𝐓
grasp
∗
=
arg
⁡
min
𝐓
grasp
(
i
)
⁡
(
λ
t
​
‖
𝐭
grasp
(
i
)
−
𝐭
ref
‖
2
+
λ
R
​
d
geo
​
(
𝐑
grasp
(
i
)
,
𝐑
ref
)
)
,
\mathbf{T}_{\mathrm{grasp}}^{\ast}=\arg\min_{\mathbf{T}_{\mathrm{grasp}}^{(i)}}\Bigl(\lambda_{t}\,\|\mathbf{t}_{\mathrm{grasp}}^{(i)}-\mathbf{t}_{\mathrm{ref}}\|_{2}+\lambda_{R}\,d_{\mathrm{geo}}\!\bigl(\mathbf{R}_{\mathrm{grasp}}^{(i)},\mathbf{R}_{\mathrm{ref}}\bigr)\Bigr),
(12)
where
λ
t
\lambda_{t}
and
λ
R
\lambda_{R}
balance translation and rotation consistency, respectively. We then insert
𝐓
grasp
∗
\mathbf{T}_{\mathrm{grasp}}^{\ast}
into the recovered EE trajectory as an intermediate target and smooth the resulting motion via interpolation.
Inverse kinematics then converts the smoothed trajectory into an executable action sequence
{
𝐚
t
}
t
=
1
N
\{\mathbf{a}_{t}\}_{t=1}^{N}
for controlling the robot arm, as illustrated in
Fig.
4
.
