The Maxwell's equations in vacuum

$$\begin{cases} \frac{\partial \mathbf{E}}{\partial t} = c \nabla \times (c\mathbf{B}) \\ \frac{\partial (c\mathbf{B})}{\partial t} = -c \nabla \times \mathbf{E} \end{cases}$$

If we know the initial field E(r,t=0) and B(r,t=0), we are able to get the evolution of the field E(r,t) and B(r,t)

One way is using Finite-difference time-domain (FDTD) method.

One way is solving the evolution of  $\textbf{\textit{E}}\$  and  $\ \textbf{\textit{B}}\$  in  $\ \textbf{\textit{k}}\$  space. 

Apply Fourier transform in  ${\it k}$  space

$$\begin{cases}
\mathbf{F}\mathbf{E}(\mathbf{k},t) = F\{\mathbf{E}(\mathbf{r},t)\} \\
\mathbf{F}\mathbf{B}(\mathbf{k},t) = F\{c\mathbf{B}(\mathbf{r},t)\}
\end{cases}$$

The Maxwell's equations will become

$$\begin{cases} \frac{\partial \mathbf{FE}}{\partial t} = ic\mathbf{k} \times \mathbf{FB} \\ \frac{\partial \mathbf{FB}}{\partial t} = -ic\mathbf{k} \times \mathbf{FE} \end{cases}$$

We can re-linearize this equation

$$\begin{cases} FA_{+} = \frac{1}{2} \left( FE - \hat{k} \times FB \right) \\ FA_{-} = \frac{1}{2} \left( FE + \hat{k} \times FB \right) \end{cases}$$

And get their evolution equations

$$\frac{\partial FA_{+}}{\partial t} = \frac{1}{2} \left( \frac{\partial FE}{\partial t} - \hat{k} \times \frac{\partial FB}{\partial t} \right) \\
= \frac{ic}{2} \left( k \times FB + \hat{k} \times (k \times FE) \right) \\
= \frac{ic}{2} \left( k \times FB + (\hat{k} \cdot FE)k - (\hat{k} \cdot k)FE \right) \\
= -\frac{ic}{2} \left( |k|FE - |k|\hat{k} \times FB \right) \\
= -ic|k|FA_{+} \\
= -i\omega FA_{+}$$

$$\begin{split} \frac{\partial FA_{-}}{\partial t} &= \frac{1}{2} \Big( \frac{\partial FE}{\partial t} + \hat{\mathbf{k}} \times \frac{\partial FB}{\partial t} \Big) \\ &= \frac{ic}{2} \Big( \mathbf{k} \times FB - \hat{\mathbf{k}} \times (\mathbf{k} \times FE) \Big) \\ &= \frac{ic}{2} \Big( \mathbf{k} \times FB - \Big( \hat{\mathbf{k}} \cdot FE \Big) \mathbf{k} + \Big( \hat{\mathbf{k}} \cdot \mathbf{k} \Big) FE \Big) \\ &= \frac{ic}{2} \Big( |\mathbf{k}| FE + |\mathbf{k}| \hat{\mathbf{k}} \times FB \Big) \\ &= ic |\mathbf{k}| FA_{-} \\ &= i\omega FA_{-} \end{split}$$

Note that the EM field in vacuum are transverse wave with  $\mathbf{k} \cdot \mathbf{F} \mathbf{E} = \mathbf{k} \cdot \mathbf{F} \mathbf{B} = 0$ .

The dispersion relation is  $\omega = c|\mathbf{k}|$ .

We can find that these 2 equations can be decoupled. 
These two variables are called Normal variables.
We can get the solution for each of them

 $FE(k,t) = FA_{\perp}(k,t) + FA_{\perp}(k,t)$ 

$$\begin{cases} \mathbf{F} \mathbf{A}_{+}(\mathbf{k}, t) = \mathbf{F} \mathbf{A}_{+}(\mathbf{k}, t = 0) \times \exp(-i\omega t) \\ \mathbf{F} \mathbf{A}_{-}(\mathbf{k}, t) = \mathbf{F} \mathbf{A}_{-}(\mathbf{k}, t = 0) \times \exp(i\omega t) \end{cases}$$

The inverse of the re-linearization is

$$\begin{cases} FE = FA_{+} + FA_{-} \\ FB = \hat{k} \times (FA_{+} - FA_{-}) \end{cases}$$

Then we can get the evolution from the spectrum

$$= FA_{+}(\mathbf{k}, t = 0) \times \exp(-i\omega t) + FA_{-}(\mathbf{k}, t = 0) \times \exp(i\omega t)$$

$$= \frac{1}{2} \Big( FE(\mathbf{k}, t = 0) - \hat{\mathbf{k}} \times FB(\mathbf{k}, t = 0) \Big) \times \exp(-i\omega t) + \frac{1}{2} \Big( FE(\mathbf{k}, t = 0) + \hat{\mathbf{k}} \times FB(\mathbf{k}, t = 0) \Big) \times \exp(i\omega t)$$

$$= FE(\mathbf{k}, t = 0) \times \cos(c|\mathbf{k}|t) + i\hat{\mathbf{k}} \times FB(\mathbf{k}, t = 0) \times \sin(c|\mathbf{k}|t)$$

$$FB(\mathbf{k}, t) = \hat{\mathbf{k}} \times (FA_{+}(\mathbf{k}, t) - FA_{-}(\mathbf{k}, t))$$

$$= \hat{\mathbf{k}} \times (FA_{+}(\mathbf{k}, t = 0) \times \exp(-i\omega t) - FA_{-}(\mathbf{k}, t = 0) \times \exp(i\omega t)$$

$$= \hat{\mathbf{k}} \times \Big( \frac{1}{2} \Big( FE(\mathbf{k}, t = 0) - \hat{\mathbf{k}} \times FB(\mathbf{k}, t = 0) \Big) \times \exp(-i\omega t) - \frac{1}{2} \Big( FE(\mathbf{k}, t = 0) + \hat{\mathbf{k}} \times FB(\mathbf{k}, t = 0) \Big) \times \exp(i\omega t)$$

$$= -\hat{\mathbf{k}} \times \Big( iFE(\mathbf{k}, t = 0) \times \sin(\omega t) + \hat{\mathbf{k}} \times FB(\mathbf{k}, t = 0) \times \cos(\omega t) \Big)$$

$$= FB(\mathbf{k}, t = 0) \times \cos(c|\mathbf{k}|t) - i\hat{\mathbf{k}} \times FE(\mathbf{k}, t = 0) \times \sin(c|\mathbf{k}|t)$$

Bringing the solution back to the equations can test their correctness.


Photons and Atoms: Introduction to Quantum Electrodynamics
Claude Cohen-Tannoudji, Jacques Dupont-Roc, Gilbert Grynberg
10.1002/9783527618422
