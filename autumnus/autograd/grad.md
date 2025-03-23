# Matrix multiplication back propagation

## Input, weights and output

```math
\mathbf{X} =
\begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1n} \\
x_{21} & x_{22} & \cdots & x_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
x_{m1} & x_{m2} & \cdots & x_{mn}
\end{bmatrix}
\quad

\mathbf{W} =
\begin{bmatrix}
w_{11} & w_{12} & \cdots & w_{1k} \\
w_{21} & w_{22} & \cdots & w_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
w_{n1} & w_{n2} & \cdots & w_{nk}
\end{bmatrix}
\quad

\mathbf{Y} = \mathbf{W} \mathbf{X} =
\begin{bmatrix}
y_{11} & y_{12} & \cdots & y_{1k} \\
y_{21} & y_{22} & \cdots & y_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
y_{m1} & y_{m2} & \cdots & y_{mk}
\end{bmatrix}
```

Input matrix $\mathbf{X}$ has shape of $m \times n$, weight matrix $\mathbf{W}$ has shape of $n \times k$, and output matrix $\mathbf{Y}$ has shape of $m \times k$. During forward pass we perform matrix multiplication of $\mathbf{X}$ and $\mathbf{W}$, so $\mathbf{Y}$ can be written as

```math

\mathbf{Y} =
\begin{bmatrix}
 y_{11} & \cdots & y_{1k} \\
y_{21} & \cdots & y_{2k} \\
\vdots & \ddots & \vdots \\
y_{m1} & \cdots & y_{mk}
\end{bmatrix} =
\begin{bmatrix}
x_{11}w_{11} + x_{12}w_{21} + ... + x_{1n}w_{n1} & \cdots & x_{11}w_{1k} + x_{12}w_{2k} + ... + x_{1n}w_{nk} \\
x_{21}w_{11} + x_{22}w_{21} + ... + x_{2n}w_{n1} & \cdots & x_{21}w_{1k} + x_{22}w_{2k} + ... + x_{2n}w_{nk} \\
\vdots & \ddots & \vdots \\
x_{m1}w_{11} + x_{m2}w_{21} + ... + x_{mn}w_{n1} & \cdots & x_{m1}w_{1k} + x_{m2}w_{2k} + ... + x_{mn}w_{nk}
\end{bmatrix},
```

this view helps to figure out how to compute gradient of $L$ (L stands for loss) with respect to input and weights.

## Gradients computation (via chain rule)

Gradient matrix of $L$  with respect to $\mathbf{Y}$ will be

```math
\begin{bmatrix}
\frac{dL}{dy_{11}} & \frac{dL}{dy_{12}} & \cdots & \frac{dL}{dy_{1k}} \\
\frac{dL}{dy_{21}} & \frac{dL}{dy_{22}} & \cdots & \frac{dL}{dy_{2k}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{dL}{dy_{m1}} & \frac{dL}{dy_{m2}} & \cdots & \frac{dL}{dy_{mk}}
\end{bmatrix} ,
```

and gradient matricies of $\mathbf{L}$ with respect to input and weights (via chain rule) are

```math

\frac{dL}{d\mathbf{X}} =
\begin{bmatrix}
\frac{dL}{dx_{11}} & \frac{dL}{dx_{12}} & \cdots & \frac{dL}{dx_{1n}} \\
\frac{dL}{dx_{21}} & \frac{dL}{dx_{22}} & \cdots & \frac{dL}{dx_{2n}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{dL}{dx_{m1}} & \frac{dL}{dx_{m2}} & \cdots & \frac{dL}{dx_{mn}}
\end{bmatrix}
\quad

\frac{dL}{d\mathbf{W}} =
\begin{bmatrix}
\frac{dL}{dw_{11}} & \frac{dL}{dw_{12}} & \cdots & \frac{dL}{dw_{1k}} \\
\frac{dL}{dw_{21}} & \frac{dL}{dw_{22}} & \cdots & \frac{dL}{dw_{2k}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{dL}{dw_{n1}} & \frac{dL}{dw_{n2}} & \cdots & \frac{dL}{dw_{nk}}
\end{bmatrix}.

```

### Gradient with respect to $\mathbf {X}$

If we look up on the $\mathbf{Y}$ computation then we can notice that each $\frac{dL}{dx_{ij}}$ is actually $\sum_{c = 1}^{k} \frac{dL}{dy_{ic}} \cdot \frac{dy_{ic}}{dx_{ij}} = \sum_{c = 1}^{k} \frac{dL}{dy_{ic}} \cdot {w_{jc}}$ so we can write it as $\frac{dL}{d\mathbf{Y}} \cdot \mathbf{W}^T$.



### Gradient with respect to $\mathbf {W}$

Same for $\frac{dL}{dW}$ we can notice that each $\frac{dL}{dw_{ij}}$ is actually $\sum_{c = 1}^{m} \frac{dL}{dy_{cj}} \cdot \frac{y_{cj}}{w_{ij}} = \sum_{c = 1}^{m} \frac{dL}{y_{cj}} \cdot x_{ci}$ so we can write it as $X^T \cdot \frac{dL}{\mathbf{Y}}$.
