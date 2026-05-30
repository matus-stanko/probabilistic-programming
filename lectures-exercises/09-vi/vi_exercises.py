# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Easy exercises
#
# 1. What structural and differentiability assumptions on the model are required for black-box ADVI? What assumptions, if any, are imposed on the data?
#
# 2. How is the guide defined in ADVI? Describe both mean-field and full-rank ADVI. Why is directly optimizing the covariance matrix $\Sigma$ problematic?
#
# 3. How do we parameterize $\Sigma$ to ensure that it stays positive definite? Consider both the mean-field and full-rank cases.
#
# 4. ADVI optimizes the evidence lower bound (ELBO). Assuming continuous observations, can the ELBO be positive? Why? If so, how should a positive ELBO be interpreted?
#
# 5. For a 300-dimensional regression model with a Gaussian likelihood, a Gaussian prior on the coefficients, and a half-normal prior on the variance, compare the following inference methods:
#    - Mean-field ADVI
#    - MAP
#    - SVGD
#    - Laplace
#
#    Rank the methods in terms of:
#    - computational complexity,
#    - ability to capture tail behavior,
#    - memory usage,
#    - accuracy in capturing the mode.
#
#    You may assume all methods are run to reasonable convergence. **Extra question:** How would your answer change if the variance prior was an inverse-gamma distribution instead?
#
# %% [markdown]
# ## Medium exercises
#
# 1. Show that SVGD with one particle and a translation-invariant kernel $k(x,y)=f(x-y)$, where $f \in C^1(\mathbb{R}^d,\mathbb{R}^+)$ is even, corresponds to MAP estimation for the posterior $p(\theta \mid \mathcal{D})$. Recall that the SVGD update equation is given in the lecture slides.
#
# 2. For SVGD, the choice of kernel can impact approximation quality. Using the two-bananas distribution from the lecture, implement the following kernels and check their effect on the SVGD particle approximation. Here, $\|\cdot\|$ denotes the Euclidean norm.
#    - IMQ kernel: $k(x,y)=(c^2 + \|x-y\|^2)^\beta$ (assume $\beta < 0$)
#    - Linear kernel: $k(x,y)=x^\top y + 1$
#    - Matérn kernel ($\nu=3/2$): $k(x,y)=\alpha^2\left(1 + \frac{\sqrt{3}\|x-y\|}{\ell}\right)\exp\left(-\sqrt{3}\frac{\|x-y\|}{\ell}\right)$ (assume $\ell>0$)
#
#    Is there a difference in mode estimation? How does the kernel choice affect tail approximation? Use the same step-size schedule, number of particles, and initialization across kernels.
#
# 3. For each of the following methods, write down the quantity being optimized (or approximated) and suggest a criterion to monitor for convergence:
#    - MAP
#    - Mean-field ADVI
#    - SVGD
#    - Laplace
# %% [markdown]
# ## Hard exercises
#
# 1. Any positive definite kernel induces a reproducing kernel Hilbert space (RKHS); however, not all kernels are well suited for SVGD. Consider the exponential (Laplacian) kernel
#    $$
#    k(x,y)=\exp\left(-\frac{\|x-y\|}{2h}\right).
#    $$
#    Explain the numerical pathology this kernel can introduce in SVGD.
#
#    *(Hint: compute $\nabla_x k(x,y)$ and consider the behavior as $x \to y$.)*
#
# 2. We briefly discussed ADVI and the Laplace approximation in the lecture. Both methods estimate the first and second moments of the posterior; however, they differ algorithmically, which affects the approximations they provide.
#
#    i. Define a model with a multimodal posterior (e.g., a two-component Gaussian mixture model) and implement it in PyMC. Approximate the posterior using both ADVI (`pymc.fit`) and Laplace (`from pymc_extras.inference import fit_laplace`).
#
#    ii. Overlay contour plots of the approximate and true posteriors. Where tractable, compute the KL divergence between the approximation and the true posterior (the KL divergence is defined in the lecture slides).
#
#    iii. What is the effect of varying initialization? How do the two methods differ in capturing the posterior? Explain the behaviour based on their algorithmic differences.

# %%
