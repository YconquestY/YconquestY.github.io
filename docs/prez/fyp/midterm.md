---
layout: SlidePage
title: ELEC4848
---
@slidestart auto

# Framework-Algorithm Co-design for Neural Rendering

## Midterm Report of ELEC4848

Yu Yue (Will)

---

## KEYWORDS

SysML, CG

--

I developed a deep learning framework **from scratch**.

It is called `Needle`.
<!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

I developed a deep learning framework **from scratch**.

```python
import needle as ndl
import needle.nn as nn
import needle.optim as optim
…
```

--

<!-- .slide: data-auto-animate -->

I developed a deep learning framework **from scratch**.

I am implementing neural rendering algorithms for *novel view synthesis*.

--

<!-- .slide: data-auto-animate -->

I am implementing neural rendering algorithms for *novel view synthesis*.

![](/prez/fyp/nvs.png)

---

## Needle

- Automatic differentiation
<!-- .element: class="fragment fade-in" -->

- CPU/CUDA backend
<!-- .element: class="fragment fade-in" -->

- Utility APIs for layers, optimizers, …
<!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

## Needle

### Automatic differentiation

*Topological sort* of computational graphs
<!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

## Needle

### CPU/CUDA backend

- Interaction of Python and C++/CUDA via [`pybind11`](https://pybind11.readthedocs.io/)
<!-- .element: class="fragment fade-in" -->

- Parallel algorithms implementation
<!-- .element: class="fragment fade-in" -->
  - Element-wise operators, reduction, scan, GeMM/GeMV, SpMV, histogram, …
  <!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

## Needle

### CPU/CUDA backend

- Half-precision training on tensor cores
  - [`tcnn`](https://github.com/NVlabs/tiny-cuda-nn) at SIGGRAPH'21

--

<!-- .slide: data-auto-animate -->

## Needle

### CPU/CUDA backend

- Sparse training
  - [Pixelfly](https://hazyresearch.stanford.edu/blog/2022-01-17-Sparsity-3-Pixelated-Butterfly) at ICLR'22<br>
  ![](/prez/fyp/pixelfly.png)
  <!-- .element: class="fragment fade-in" -->

--

- Utility APIs for layers, optimizers, …
  - Convolution
  <!-- .element: class="fragment fade-in" -->
    - Forward pass
    <!-- .element: class="fragment fade-in" -->
      - im2col & GEMM
      <!-- .element: class="fragment fade-in" -->
      - Winograd
      <!-- .element: class="fragment fade-in" -->
      - Fast Fourier transform
      <!-- .element: class="fragment fade-in" -->
    - Backprop: gradient of convolution?
    <!-- .element: class="fragment fade-in" -->
  - LSTM
  <!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

## Needle

"Talk is cheap. Show me the code." ― Linus Torvalds

--

<!-- .slide: data-auto-animate -->

## Needle

https://github.com/YconquestY/Needle

---

## NVS

![](/prez/fyp/nvs.png)

<!-- .element: class="r-stretch" -->

--

<!-- .slide: data-auto-animate -->

## NVS

![](/prez/fyp/3d_supervision.png)

<!-- .element: class="r-stretch" -->

--

<!-- .slide: data-auto-animate -->

## NVS

![](/prez/fyp/2d_supervision.png)

<!-- .element: class="r-stretch" -->

--

<!-- .slide: data-auto-animate -->

## NVS

My [blog](https://yconquesty.github.io/blog/ml/nerf/nerf_rendering.html) on differentiable rendering

![](/prez/fyp/volume_rendering.png)

<!-- .element: class="r-stretch" -->

--

<!-- .slide: data-auto-animate -->

## NVS

![](/prez/fyp/implicit_representation.png)

<!-- .element: class="r-stretch" -->

This is *neural radiance field* (NeRF) at ECCV'20.
<!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

## NVS

![](/prez/fyp/hybrid_representation.png)

<!-- .element: class="r-stretch" -->

--

<!-- .slide: data-auto-animate -->

## NVS

![](/prez/fyp/hybrid_forms.png)

<!-- .element: class="r-stretch" -->

Representatives: NSVF (NeurIPS'20), plenOctrees (ICCV'21), plenoxels (CVPR'22), point-NeRF (CVPR'22), instant-NGP (SIGGRAPH'22), and TensoRF (ECCV'22)
<!-- .element: class="fragment fade-in" -->

---

## NVS on Needle

### Customized CUDA kernels

Based on TensoRF

- Bilinear interpolation
<!-- .element: class="fragment fade-in" -->
- Tensor decomposition
<!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

## NVS on Needle

### Kernel fusion

Given a ray $\boldsymbol{r} = \boldsymbol{o} + z\boldsymbol{d}$, the pixel color is

$$
\mathbf{C}(\boldsymbol{r})\
=\
\int_{z_n}^{z_f} T(z) \sigma \left( \boldsymbol{r}(z) \right) \boldsymbol{c} \left(\boldsymbol{r}(z), \boldsymbol{d} \right) \ dz
$$

where $T(z) =  \exp \left(-\int_{z_n}^z \sigma \left(\boldsymbol{r} (s) \right) \ ds \right)$

--

<!-- .slide: data-auto-animate -->

## NVS on Needle

### Kernel fusion

$\boldsymbol{r} = \boldsymbol{o} + z\boldsymbol{d}$, $z$ ranges from $z_1$ to $z_N$

$$
\hat{\mathbf{C}}(\boldsymbol{r})\
=\
\sum_{i=1}^{N} T_i \left(1 - e^{-\sigma_i \delta_i} \right) \boldsymbol{c}_i
$$

where $T_i = \exp \left(-\sum_{j=1}^{i-1} \sigma_j \delta_j \right)$

--

<!-- .slide: data-auto-animate -->

## NVS on Needle

### Kernel fusion

- Accelerate backpropagation
<!-- .element: class="fragment fade-in" -->
- Minimize global memory access and kernel launch overhead
<!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

## NVS on Needle

Code will be released on GitHub in due time.

---

## SUMMARY

I developed a deep learning framework **from scratch**. The code is available at https://github.com/YconquestY/Needle.

- Extensive efforts on kernel optimization
<!-- .element: class="fragment fade-in" -->
- About $6$k lines available
<!-- .element: class="fragment fade-in" -->
- Remaining $4$k lines in progress
<!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

## SUMMARY

I am implementing neural rendering algorithms for NVS. My blog series are

1. [A Surge in NeRF](https://yconquesty.github.io/blog/ml/nerf/)
2. [NeRF: A Volume Rendering Perspective](https://yconquesty.github.io/blog/ml/nerf/nerf_rendering.html)
3. [NeRF: How Spherical Harmonics Works](https://yconquesty.github.io/blog/ml/nerf/nerf_sh.html)
4. [NeRF: How NDC Works](https://yconquesty.github.io/blog/ml/nerf/nerf_ndc.html)

- About $15$k words available
<!-- .element: class="fragment fade-in" -->
- About $7$k words in progress
<!-- .element: class="fragment fade-in" -->

---

## Q & A

Why **isn't** there a demo for Needle or NeRF?

- Limited presentation time
<!-- .element: class="fragment fade-in" -->
- See my GitHub repositories!
<!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

## Q & A

Is it useful to implement an ML framework from scratch? Why not PyTorch?

No.
<!-- .element: class="fragment fade-out" -->

- "Re-inventing the wheels" is common in the systems field.
<!-- .element: class="fragment fade-in" -->
- Production graphics system is more about engineering.
<!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

## Q & A

What then?

ML compilers!

--

<!-- .slide: data-auto-animate -->

## Q & A

What then?

Key problems in SysML:

- Automatic parallelism
<!-- .element: class="fragment fade-in" -->
- Code generation
<!-- .element: class="fragment fade-in" -->

Which ML infrastructre is to blame, scheduler, framework, or compiler?
<!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

## Q & A

What then?

NeRF is suitable for new papers, but not for production, at least from my perspective, in that the algorithm "family" is far from computationally efficient in nature.

[Luma AI](https://lumalabs.ai) is the spearhead in commercializing NVS, but cloud GPU costs much.

This is a beginning, rather than an end, of neural rendering!
<!-- .element: class="fragment fade-in" -->

--

<!-- .slide: data-auto-animate -->

## Q & A

Further questions?

---

## Thank you.

@slideend
