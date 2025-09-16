# WGAN-GP algorithm

``` latex
\begin{algorithm}[H]
    \caption{Training loop for critic GAN with gradient penalty (non-batched)}
    \begin{algorithmic}[1]
    \State \textbf{Inputs:} critic steps $n_{\mathrm{critic}}$, gradient penalty weight $\lambda$
    \State Optimisers: $\mathrm{Optimiser}_D$, $\mathrm{Optimiser}_G$
    \State Initialise parameters $\theta_D, \theta_G$
    \For{epoch = 1 \textbf{to} num\_epochs}
        \For{$i = 1$ \textbf{to} $n_{\mathrm{critic}}$}
          \State $\mathbf{x}(t)\sim p_{\mathrm{real}}$ \Comment Sample real trajectory
          \State $\mathbf{z}_0(t)\sim p_\mathrm{noise}$ \Comment Sample seed
          \State $\mathbf{z}(t)\gets G_{\theta_G}\bigl(\mathbf{z}_0(t)\bigr)$ \Comment Compute fake trajectory
          \State $\varepsilon\sim \mathcal U(0,1)$ \Comment Sample interpolation parameter
          \State $\mathbf{y}(t) \;\gets\; \varepsilon\,\mathbf{x}(t)\;+\;(1-\varepsilon)\,\mathbf{z}(t)$ \Comment Interpolate
          \State $f_{\mathrm{real}} \;\gets\; f_{\theta_D}\bigl(\mathbf{x}(t)\bigr)$ \Comment Evaluate critic scores
          \State $f_{\mathrm{fake}} \;\gets\; f_{\theta_D}\bigl(\mathbf{z}(t)\bigr)$ \Comment Evaluate critic scores
          \State $\mathcal C_{\mathrm{GP}} \;=\; \lambda\,\bigl(\|\nabla_{\mathbf{y}}\,f_{\theta_D}\bigl(\mathbf{y}(t)\bigr)\|_2 - 1\bigr)^2$ \Comment Compute gradient penalty
          \State $\mathcal C_D \;=\; f_{\mathrm{fake}} - f_{\mathrm{real}} + \mathcal C_{\mathrm{GP}}$ \Comment Compute critic cost (to minimise)
          \State $\theta_D \;\gets\; \mathrm{Optimiser}_D.\mathrm{step}(\theta_D,\;\nabla_{\theta_D}\,\mathcal C_D)$ \Comment Update critic parameters
        \EndFor
    
        \State $\mathbf{z}_0(t)\sim p_z$ \Comment Sample seed
        \State $\mathbf{z}(t)\gets G_{\theta_G}\bigl(\mathbf{z}_0(t)\bigr)$ \Comment Compute fake trajectory
        \State $f_{\mathrm{fake}} \;\gets\; f_{\theta_D}\bigl(\mathbf{z}(t)\bigr)$ \Comment Evaluate critic scores
        \State $\mathcal C_G \;=\; -\,f_{\mathrm{fake}}$ \Comment Compute generator cost
        \State $\theta_G \;\gets\; \mathrm{Optimiser}_G.\mathrm{step}(\theta_G,\;\nabla_{\theta_G}\,\mathcal C_G)$ \Comment Update generator parameters
    \EndFor
    \end{algorithmic}
    \end{algorithm}
```
