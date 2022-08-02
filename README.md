# Bidomain-Model
The bidomain model provides insights into the electrophysiology of the myocardium by modeling potential waves using a system of partial differential equations (PDEs). Consider the bounded domain $\Omega \subset \mathbb{R}^n$ representing the myocardium. We formulate the system as follows:
```math
\begin{aligned}
  \partial_t u - \frac{1}{\varepsilon}f(u,v) &- \nabla \cdot (\sigma_i \nabla (u+u_e)) =0\\
   \nabla \cdot (\sigma_i \nabla u &+(\sigma_i + \sigma_e)\nabla u_e)  =0\\
	\partial_t v &- \varepsilon g(u,v)=0
\end{aligned}
```
with homogeneous Neumann boundary conditons ( $\nabla u \cdot\vec{n}= \nabla u_e \cdot\vec{n}=0 \text{ on } \partial\Omega$ ) and where
```math
u=u_i-u_e 
```
 is the transmembrane potential, the difference between the intracellular and extracellular potential, $\sigma_i,\sigma_e$ are second order tensors ($2x2$ matrices) representing the electrical conductivity in each spatial direction, $v$ is a lumped ionic variable, $\varepsilon$ is the ratio between the repolarisation rate and tissue excitation rate.
$f$ and $g$ are functions representing the ionic activity in the myocardium. This ionic activity is incorporated into our approach with the 3rd equation, and is not typically part of the bidomain model. We are using the FitzHugh-Nagumo equations, which are more adapted to the heart muscle than other models and write $f$ and $g$ as
```math
\begin{aligned}
f(u,v)&= u-\frac{u^3}{3}-v\\
g(u,v)&=u+\beta -\gamma v.
\end{aligned}
```
where $\gamma$ is a parameter controlling the ion transport and $\beta$ represents the cell excitability.


The Julia file is written for/in Pluto.jl.
