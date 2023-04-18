---
layout: Slide
---
@slidestart auto

# Framework-Algorithm Co-design for Neural Rendering

Yu Yue

---

## Keywords

SysML, CG, and HPC

---

## Outline

- Needle: a DL framework from scratch
- NVS with "NeRF"
  - A fused kernel for neural rendering
- Summary
- Q & A

---

![](/prez/fyp/perceptron.png)

<!-- .element: class="r-stretch" -->

This is a *perceptron*.

--

<!-- .slide: data-auto-animate -->

![](/prez/fyp/perceptron.png)

<!-- .element: class="r-stretch" -->

```python
import torch
import torch.nn as nn

perceptron = nn.Sequential(nn.Linear(…),
                           nn.Sigmoid())
```

--

<!-- .slide: data-auto-animate -->

```python
import torch
import torch.nn as nn

perceptron = nn.Sequential(nn.Linear(…),
                           nn.Sigmoid())
```

Given batched input $\mathbf{X}$, the forward pass is

$$
\mathbf{Y} = \sigma(\mathbf{X} \mathbf{W} + \boldsymbol{b})
$$

--

<!-- .slide: data-auto-animate -->

$$
\mathbf{Y} = \sigma(\mathbf{X} \mathbf{W} + \mathbf{b})
$$

![](/prez/fyp/perceptron_compute_graph.png)

--

<!-- .slide: data-auto-animate -->

![](/prez/fyp/perceptron_compute_graph.png)

```python
import needle as ndl
import needle.nn as nn

perceptron = nn.Sequential(nn.Linear(…),
                           nn.Sigmoid())
```

--

<!-- .slide: data-auto-animate -->

```python
import needle as ndl
import needle.nn as nn

perceptron = nn.Sequential(nn.Linear(…),
                           nn.Sigmoid())
```

![](/prez/fyp/needle_arch.png)

--

![](/prez/fyp/leaf_node.png)

Gradient descent
<!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

Gradient descent

$$
\frac{d~\text{Sigmoid}(x)}{dx}
=
\frac{e^{-x}}{(1 + e^{-x})^2}
$$
<!-- .element: class="fragment fade-in-then-semi-out" -->

$$
\frac{d~\text{Linear}}{d \mathbf{W}}
=
\mathbf{X}^\mathsf{T}
$$
<!-- .element: class="fragment fade-in-then-semi-out" -->

$$
\frac{d~\text{Linear}}{d\boldsymbol{b}}
=
\boldsymbol{1}
$$
<!-- .element: class="fragment fade-in-then-semi-out" -->

$$
\frac{d~(f \circ g)}{dx}
=
\frac{df}{dg}
\frac{dg}{dx}
$$
<!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

$$
\frac{d~(f \circ g)}{dx}
=
\frac{df}{dg}
\frac{dg}{dx}
$$

```python
convnet = nn.Sequential(nn.Conv(…),
                        nn.ReLU(),
                        nn.Conv(…),
                        nn.ReLU(),
                        nn.Linear(),
                        nn.ReLU())
```

--

<!-- .slide: data-auto-animate -->

```python
convnet = nn.Sequential(nn.Conv(…),
                        nn.ReLU(),
                        nn.Conv(…),
                        nn.ReLU(),
                        nn.Linear(),
                        nn.ReLU())
```

Automatic differentiation

--

<!-- .slide: data-auto-animate -->

Automatic differentiation

![](/prez/fyp/backprop_1.png)

--

![](/prez/fyp/backprop_2.png)

--

![](/prez/fyp/backprop_3.png)

--

![](/prez/fyp/backprop_4.png)

--

<!-- .slide: data-auto-animate -->

Automatic differentiation

![](/prez/fyp/leaf_node.png)

Reverse traversal; gradient cumulated
<!-- .element: class="fragment fade-in" -->

--

Reverse mode Autodiff

Given output $y$ and input $v_i$, *adjoint*: $\bar{v}_i := \frac{\partial y}{\partial v_i}$.

![](/prez/fyp/autograd.png)

<!-- .element: class="r-stretch" -->

--

![](/prez/fyp/autograd_sort.png)

<!-- .element: class="r-stretch" -->

--

![](/prez/fyp/autograd_op.png)

<!-- .element: class="r-stretch" -->

--

![](/prez/fyp/autograd_input.png)

<!-- .element: class="r-stretch" -->

--

![](/prez/fyp/autograd_grad.png)

<!-- .element: class="r-stretch" -->

--

Needle `Tensor`

![](/prez/fyp/needle_tensor_autograd.png)

<!-- .element: class="r-stretch" -->

--

Needle `Op`

![](/prez/fyp/needle_op.png)

<!-- .element: class="r-stretch" -->

- What is the gradient of broadcasting?
- What is the gradient of dilated convolution?

<!-- .element: class="fragment fade-in" -->

--

Needle `Tensor`

![](/prez/fyp/needle_tensor_ndarray.png)

<!-- .element: class="r-stretch" -->

--

An $n$-dimensional array

![](/prez/fyp/needle_ndarray.png)

<!-- .element: class="r-stretch" -->

--

Live demo!

![](/prez/fyp/needle_arch.png)

--

Training a ConvNet

![](/prez/fyp/ResNet9.png)

<!-- .element: class="r-stretch" -->

--

Training an LSTM

![](/prez/fyp/rnn.png)

<!-- .element: class="r-stretch" -->

---

The *novel view synthesis* problem

![](/prez/fyp/nvs.png)

--

![](/prez/fyp/volume_rendering.png)

--

![](/prez/fyp/implicit_representation.png)

--

![](/prez/fyp/hybrid_representation.png)

--

<!-- .slide: data-auto-animate -->

![](/prez/fyp/hybrid_forms.png)

Neural mesh: MobileNeRF (ECCV'22)
<!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

![](/prez/fyp/hybrid_forms.png)

Neural point cloud: PointNeRF (CVPR'22)

--

<!-- .slide: data-auto-animate -->

![](/prez/fyp/hybrid_forms.png)

Multi-resolution hash: Instant NGP (SIGGRAPH'22)

--

<!-- .slide: data-auto-animate -->

![](/prez/fyp/hybrid_forms.png)

Neural codebook: VQ-RF (SIGGRAPH'22)

--

<!-- .slide: data-auto-animate -->

![](/prez/fyp/hybrid_forms.png)

Voxel grid: Plenoxels (CVPR'22), TensoRF (ECCV'22)

--

<!-- .slide: data-auto-animate -->

![](/prez/fyp/scene.png)

<!-- .element: class="r-stretch" -->

--

<!-- .slide: data-auto-animate -->

![](/prez/fyp/scene.png)

<!-- .element: class="r-stretch" -->

$$
\boldsymbol{c}
=
\mathcal{G}_{\boldsymbol{c}}(\boldsymbol{x}, \boldsymbol{d})
\text{,}~\mathcal{G}_\boldsymbol{c} \in \mathbb{R}^4
$$

$$
\sigma
=
\mathcal{G}_\sigma(\boldsymbol{x})
\text{,}~\mathcal{G}_\sigma \in \mathbb{R}^3
$$

--

<!-- .slide: data-auto-animate -->

Tensor decomposition!

![](/prez/fyp/tensor_decomposition.png)

<!-- .element: class="r-stretch" -->

--

<!-- .slide: data-auto-animate -->

![](/prez/fyp/tensor_decomposition.png)

<!-- .element: class="r-stretch" -->

$$
\mathcal{T}
=
\sum_{r=1}^R \boldsymbol{v}_r^1 \circ \boldsymbol{v}_r^2 \circ \boldsymbol{v}_r^3
$$

--

<!-- .slide: data-auto-animate -->

![](/prez/fyp/tensor_decomposition.png)

<!-- .element: class="r-stretch" -->

$$
\mathcal{T}
=
\sum_{r=1}^{R_1} \boldsymbol{v}_r^1 \circ \mathbf{M}^{2,3}_r
+
\sum\_{r=1}^{R_2} \boldsymbol{v}_r^2 \circ \mathbf{M}^{1,3}_r
+
\sum\_{r=1}^{R_3} \boldsymbol{v}_r^3 \circ \mathbf{M}^{1,2}_r
$$

--

<!-- .slide: data-auto-animate -->

I <font color="red">failed</font> in porting TensoRF from PyTorch to Needle.

- OOM error
<!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

I <font color="red">failed</font> in porting TensoRF from PyTorch to Needle.

- Needle produces a new tensor for each operation, including indexing a tensor.
  <!-- .element: class="fragment fade-in" -->
- TensoRF and its follow-ups index tensors a lot.
  <!-- .element: class="fragment fade-in" -->
- Too many tensors for an RTX3090 to hold.
  <!-- .element: class="fragment fade-in" -->
- Distributed training? NCCL is a headache.
  <!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

Warp an unbounded $360$° scene to a "ball"

$$
\boldsymbol{x}
\longrightarrow
\boldsymbol{x} \text{ if } \|\boldsymbol{x}\| \le 1
$$

--

<!-- .slide: data-auto-animate -->

Warp an unbounded $360$° scene to a "ball"

$$
\boldsymbol{x}
\longrightarrow
\left(2 - \frac{1}{\|\boldsymbol{x}\|}\right) \frac{\boldsymbol{x}}{\|\boldsymbol{x}\|}~\text{otherwise}
$$

--

Conversion from a "ball" to a "box"

$$
\boldsymbol{x} \color{gray}{\in \mathbb{R}^3}
\longrightarrow
\boldsymbol{x}'
\text{ such that }
x_i
=
\frac{x_i \|\boldsymbol{x}\|}{\max(|x_1|, |x_2|, |x_3|)}
$$

--

Scene warping

- Scene expressiveness $\uparrow$
  <!-- .element: class="fragment fade-in" -->
- Image quality $\downarrow$
  <!-- .element: class="fragment fade-in" -->

--

A blog series on NeRF

1. [A Surge in NeRF](https://yconquesty.github.io/blog/ml/nerf/)
2. [NeRF: A Volume Rendering Perspective](https://yconquesty.github.io/blog/ml/nerf/nerf_rendering.html)
3. [NeRF: How Spherical Harmonics Works](https://yconquesty.github.io/blog/ml/nerf/nerf_sh.html)
4. [NeRF: How NDC Works](https://yconquesty.github.io/blog/ml/nerf/nerf_ndc.html)

--

Highly positive feedback

![](/prez/fyp/feedback.jpg)

<!-- .element: class="r-stretch" -->

---

<!-- .slide: data-auto-animate -->

Ray $\boldsymbol{r} = \boldsymbol{o} + z\boldsymbol{d}$ with $z$ ranges from $z_1$ to $z_N$

$$
\hat{\mathbf{C}}(\boldsymbol{r})
=
\sum_{i=1}^{N} T_i \left(1 - e^{-\sigma_i \delta_i} \right) \boldsymbol{c}_i
, \
T_i = \exp \left(-\sum\_{j=1}^{i-1} \sigma_j \delta_j \right)
$$

--

<!-- .slide: data-auto-animate -->

$$
\hat{\mathbf{C}}(\boldsymbol{r})
=
\sum_{i=1}^{N} \color{red}{T_i} \color{green}{\left(1 - e^{-\sigma_i \delta_i} \right)} \color{blue}{\boldsymbol{c}_i}
, \
T_i = \exp \left(-\sum\_{j=1}^{i-1} \sigma_j \delta_j \right)
$$

Elementwise ternary operation
<!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

$$
\hat{\mathbf{C}}(\boldsymbol{r})
=
\color{blue}{\sum_{i=1}^{N}} T_i \left(1 - e^{-\sigma_i \delta_i} \right) \boldsymbol{c}_i
, \
T_i = \exp \left(-\sum\_{j=1}^{i-1} \sigma_j \delta_j \right)
$$

Sum reduction
<!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

$$
\hat{\mathbf{C}}(\boldsymbol{r})
=
\sum_{\color{blue}{i}=1}^{N} T_\color{blue}{i} \left(1 - e^{-\sigma_i \delta_i} \right) \boldsymbol{c}_i
, \
T_\color{blue}{i} = \exp \left(-\sum\_{j=1}^{\color{blue}{i-1}} \sigma_j \delta_j \right)
$$

Prefix sum
<!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

Kernel fusion!

Reduction

- To minimize control divergence
- To minimize memory divergence
- To minimize global memory access
- Hierarchical reduction for arbitrary samples per ray
- Thread coarsening

<!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

Kernel fusion!

Scan

- The Kogge-Stone algorithm
- The Brent-Kung algorithm
- Segmented scan for arbitrary samples per ray
- Single-pass scan for memory access efficiency
- Thread coarsening

<!-- .element: class="fragment fade-in" -->

---

Summary

- I implemented a DL framework. The code is open source.
<!-- .element: class="fragment fade-in" -->
- I failed in porting TensoRF from PyTorch to Needle. It's emberrassing…the code will not be released.
<!-- .element: class="fragment fade-in" -->
- I am maintaining a blog series on NeRF. Feedback was highly positive.
<!-- .element: class="fragment fade-in" -->
- I am working on a fused rendering kernel.
<!-- .element: class="fragment fade-in" -->

---

<!-- .slide: data-auto-animate -->

## Q & A

There seems to be much. What's the focus?

--

<!-- .slide: data-auto-animate -->

## Q & A

---

## Thank you!
