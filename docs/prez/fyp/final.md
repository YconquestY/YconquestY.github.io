---
layout: Slide
---
@slidestart auto

# Framework-Algorithm Co-design for Neural Rendering

## Final Presentation of ELEC4848

Yu Yue (Will)

---

## Keywords

System for machine learning, computer graphics

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

<!-- .element: class="r-stretch" -->

--

Gradient descent?

$$
\begin{align*}
\frac{d~\text{Sigmoid}(x)}{dx}
&=
\frac{e^{-x}}{(1 + e^{-x})^2} \\
\\
\frac{d~\text{Linear}}{d~\mathbf{W}}
&=
\mathbf{X}^\mathsf{T} \\
\frac{d~\text{Linear}}{d\boldsymbol{b}}
&=
\boldsymbol{1}
\end{align*}
$$
<!-- .element: class="fragment fade-in" -->

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


