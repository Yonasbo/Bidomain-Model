### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 60941eaa-1aea-11eb-1277-97b991548781
begin 
    using PlutoUI,HypertextLiteral, ExtendableGrids, VoronoiFVM, PlutoVista,GridVisualize,LinearAlgebra,JSON3,JLD2,DifferentialEquations,Sundials,ExtendableSparse,IncompleteLU,AlgebraicMultigrid,BenchmarkTools,DataFrames,CSV
	default_plotter!(PlutoVista);
end

# ╔═╡ b86d278f-2317-4247-92e9-50016d43ace5
html"""
<b> Scientific Computing TU Berlin Winter 2021/22  &copy; Yonas Bokredenghel </b>
<br><b> Bidomain Model</b>
"""


# ╔═╡ ab85244a-a833-4a0f-a073-0ec1f5d86cd7
md"""
# Bidomain Model
"""

# ╔═╡ db44c771-04ad-4235-a74c-9608995d93fb
TableOfContents(title="")

# ╔═╡ 087a9eac-54ad-4b3e-8479-ddf5b92dca35
md"""
## Background
"""

# ╔═╡ 95a69409-90bd-44bf-b0d9-eb6825e70c02
md"""
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
"""

# ╔═╡ c0162030-2b79-4bf7-976c-f02b65b7f98f
md"""
## Discretization
"""

# ╔═╡ 0f528d1d-2e1f-4260-928a-5483f23a2169
md"""
In this section we describe how we are discretizing space and time, leveraging the finite volume method. We will use the $\textit{method of lines}$ technique, which  first discretizes the space, resulting in a huge ODE system, and then discretizes time.
"""

# ╔═╡ 80892b19-8b1a-49f8-a6b2-8e9c19f1342a
md"""
### Space Discretization
"""

# ╔═╡ 7f8705c3-a457-4c1c-8581-e1b56e0269c1
md"""
We cover the domain $\Omega$ with a regular partition of simplexes $\omega_k$, edges in 1D, and triangles in 2D, using the SimplexGrid package, with maximal diameter $h$ and $N$ callocation points or cells. Further we suppose that we have homogeneuos Neumann boundary conditions for $u$ and $u_e$ on $\partial\Omega$.

Then, for the first equation in our system, it follows for one control volume $\omega_k$

```math
\begin{aligned}
  0&=\int_{\omega_k} \partial_t u - \frac{1}{\varepsilon}f(u,v) - \nabla \cdot (\sigma_i \nabla (u+u_e))\; d\omega  \\
   &=\int_{\omega_k} \partial_t u \; d\omega - \int_{\omega_k}\frac{1}{\varepsilon}f(u,v)\; d\omega - \int_{\omega_k}\nabla \cdot (\sigma_i \nabla (u+u_e))\; d\omega  \\
   &=\int_{\omega_k} \partial_t u \; d\omega - \int_{\omega_k}\frac{1}{\varepsilon}f(u,v)\; d\omega - \int_{\partial\omega_k} \sigma_i \nabla (u+u_e) \cdot \vec{n}_{\omega_k} \; ds  \\
   
   &\approx |\omega_k| (\partial_t u_k - \frac{1}{\varepsilon} f(u_k,v_k)) - \sum_{l \in N_k} D_{i,kl} \frac{|\delta_{kl}|}{h}((u+u_e)_k-(u+u_e)_l) \\
    
\end{aligned}
```
where $N_k$ is the set of all indices of control volumes bounding $\omega_k$, $u_k=u(x_k)$, $\delta_{kl}$ is the boundary between $\omega_k$ and $\omega_l$, and
```math
\begin{aligned}
	D_{i,kl}= \frac{\| \sigma_i (x_k-x_l)\|_2}{\| (x_k-x_l)\|_2}.
\end{aligned}
```
Since we do not have a uniform diffusion constant in the 2D case, we have diffusion constants for each spatial direction, represented in a 2-tensor. We therefore have to calculate a specific diffusion constant for each edge.

This can be written in matrix form as follows:
```math
\begin{aligned}
0 = M \partial_t \textbf{u} - \frac{1}{\varepsilon} M f(\textbf{u},\textbf{v})-A_i(\textbf{u} + \textbf{u}_e) 
\end{aligned}
```
where $\textbf{u} = [u_1,.....,u_N]^T$, same goes for $\textbf{v}$ and $\textbf{u}_e$, and $M=(m_{kl})$ and $A_i=(a_{i,kl})$ with
```math
\begin{aligned}
m_{kl} = \left\{\begin{array}{ll} |\omega_k|, & k=l \\
         0, & else \end{array}\right. ,
\end{aligned}
```
and
```math
\begin{aligned}
a_{i,kl} = \left\{\begin{array}{ll} \sum_{l' \in N_k} D_{i,kl'} \frac{|\delta_{kl'}|}{h}, & k=l \\ -D_{i,kl} \frac{|\delta_{kl}|}{h} , & l \in N_k \\
         0, & else \end{array}\right. .
\end{aligned}
```
For the other two equations follows in the same way
```math
\begin{aligned}
A_i \textbf{u} + (A_i+A_e)\textbf{u}_e &=0 \\
M \partial_t \textbf{v} - \varepsilon (M(\textbf{u}-\gamma\textbf{v})+\beta [|\omega_1|,...,|\omega_N|]^T ) &=0.
\end{aligned}
```
"""

# ╔═╡ 9e4722e2-1ca3-4758-a397-806a49c0df7a
md"""
### Time Discretization
"""

# ╔═╡ 74d89019-bad6-49c3-950b-9ed1742bbba7
md"""
For our time discretization we use a constant time step $\tau^n =\Delta t$ and, since we use the backward euler scheme to solve this, we can rewrite the matrix form in the following way
```math
\begin{aligned}
0 &=M \frac{\textbf{u}^{\textbf{n}}-\textbf{u}^{\textbf{n-1}}}{\tau^n}- \frac{1}{\varepsilon} Mf(\textbf{u}^{n},\textbf{v}^{n}) - A_i(\textbf{u}^{n}+\textbf{u}_e^{n}) \\
0 &= A_i \textbf{u}^{n} + (A_i+A_e)\textbf{u}_e^{n}  \\
0 &=M \frac{\textbf{v}^{\textbf{n}}-\textbf{v}^{\textbf{n-1}}}{\tau^n} - \varepsilon (M(\textbf{u}^{n}-\gamma\textbf{v}^{n})+\beta [|\omega_1|,...,|\omega_N|]^T ).
\end{aligned}
```
Observe that the second equation does not have a time derivative. Therefore the system reads as a Differential-Algebraic system of equations (DAE). Further $u_e$ does only occur in gradient form, and because we have Neumann boundary conditions on the domain the system of equations become singular. Hence we have to fix $u_e$ at a specific value on a particular node for every time step.
"""

# ╔═╡ 87f4dbfb-890a-41c6-8b11-3e8d6ba14ee9
md"""
## Model Implementation
"""

# ╔═╡ 788c3001-c6a0-4de0-aa3a-cdaa7dcc4c26
md"""
Here we define constant $\varepsilon$ to scale our reaction functions ($f$ and $g$) and $\gamma$ and $\beta$ which are constants of $g$.
"""

# ╔═╡ 7d7f31b2-f2e4-47da-ac4a-a67833649bcf
begin
	ε=.15 # scale reaction functions
	γ=0.5 # constant for reaction function g
	β=1.0 # constant for reaction function g
end

# ╔═╡ 7ff84bfd-9245-4f47-ad95-127904a1fa26
md"""
Create a storage function for our time derivatives $\partial_t u$ and $\partial_t v$.
"""

# ╔═╡ 1e9cadb3-0ee1-46cb-ba9e-39dfcf91f6ae
function storage(f,u,node,data)
	f[1]=u[1]
	f[2]=0
	f[3]=u[3]
end

# ╔═╡ 1a474e89-b240-4649-854a-acb10dc8f905
md"""
Define the two diffusion functions. For the 1D and 2D case.
"""

# ╔═╡ 22d57dfe-a6b5-48c8-9452-4114a30a85b0
function diffusion(f,u,data,edge)
	σ_i=1.0
	σ_e=1.0
	f[1]=σ_i*(u[1,1]-u[1,2]+u[2,1]-u[2,2])
	f[2]=(σ_i*(u[1,1]-u[1,2])+(σ_i+σ_e)*(u[2,1]-u[2,2]))
end

# ╔═╡ c0bd0f63-cb5c-4b1f-8aad-4de419182574
md"""
With the help of the edge data, we can calculate edge specific diffusion constants for the 2D case. We have two different ways of solving, one with looking at the angle between our edge and X-axes, and the other looking at the stretch factor of the Matrix, similar to the natural Matrixnorm. 
"""

# ╔═╡ 910bb2da-67f6-4661-90f6-9ad1e23d7524
function angle(a, b)
    return acosd(clamp(a⋅b/(norm(a)*norm(b)), -1, 1))
end

# ╔═╡ 9f981a30-3505-4d4a-9908-fa9499b0d1cc
function pellipse(σ_1,σ_2,ϕ)
	return sqrt(σ_1^2*cosd(ϕ)^2+σ_2^2*sind(ϕ)^2)
end

# ╔═╡ c0a55467-ce0d-4093-a87a-d88ac402c683
function diffusion_2d(f,u,edge,data)
	
	σ_i=25*Diagonal([ 0.263, 0.0263])
	σ_e=25*Diagonal([ 0.263, 0.1087])
	vec_1=vec([edge[1,1] edge[2,1]])
	vec_2=vec([edge[1,2] edge[2,2]])
	
	D_i=norm(σ_i*(vec_1-vec_2))/norm(vec_1-vec_2)
	D_e=norm(σ_e*(vec_1-vec_2))/norm(vec_1-vec_2)
	#ϕ=angle((vec_1-vec_2),vec([1 0]))
	#D_i=pellipse(25*0.263,25*0.0263,ϕ)
	#D_e=pellipse(25*0.263,25*0.1087,ϕ)
	
	f[1]=D_i*(u[1,1]-u[1,2]+u[2,1]-u[2,2])
	f[2]=(D_i*(u[1,1]-u[1,2])+(D_i+D_e)*(u[2,1]-u[2,2]))
end

# ╔═╡ 1cf82863-6529-4f05-a152-7415b0837041
function diffusion_3d(f,u,edge,data)
	
	σ_i=25*Diagonal([ 0.263, 0.0263, 0.0263])
	σ_e=25*Diagonal([ 0.263, 0.1087, 0.1087])
	vec_1=vec([edge[1,1] edge[2,1] edge[3,1]])
	vec_2=vec([edge[1,2] edge[2,2] edge[3,2]])
	
	D_i=norm(σ_i*(vec_1-vec_2))/norm(vec_1-vec_2)
	D_e=norm(σ_e*(vec_1-vec_2))/norm(vec_1-vec_2)
	
	f[1]=D_i*(u[1,1]-u[1,2]+u[2,1]-u[2,2])
	f[2]=(D_i*(u[1,1]-u[1,2])+(D_i+D_e)*(u[2,1]-u[2,2]))
end

# ╔═╡ ccbd19e6-4dbc-4f18-b510-64a8ef9b63a3
md"""
Two reaction functions $f$ and $g$.
"""

# ╔═╡ b99bd145-a064-4837-bcf4-f320f4f22426
function reaction(f,u,node,data)
    f[1]= -1/ε*(u[1]-u[1]^3/3-u[3])
    f[3]= -ε*(u[1]+β-γ*u[3])
end

# ╔═╡ cb4d82c6-e69d-4b06-baf7-d28a0951adce
md"""
Neumann Boundary conditions and fixing of $u_e(0)$ to zero at every time step by setting a dirichlet conditions just for this "species" at node $0$. Again we have two different formulations for the 1D and 2D case. 
"""

# ╔═╡ 10e2756b-f23c-4f1a-8e2a-7016d46e5d69
function bc_1d(y,u,node,data)
	if node[1]==0 
		boundary_dirichlet!(y,u,node; species=2,value=0.0)
	end
	if data[1]=="nm"
		boundary_neumann!(y,u,node; species=1,value=data[2])
		boundary_neumann!(y,u,node; species=2,value=data[3])
	elseif data[1]=="dl"
		boundary_dirichlet!(y,u,node; species=1,value=data[2])
		boundary_dirichlet!(y,u,node; species=2,value=data[3])
	end
end;

# ╔═╡ a8f21213-8866-4fdd-b48d-aec531bd9200
function bc_2d(y,u,node,data)
	
	if node[1]==0 && node[2]==0
		boundary_dirichlet!(y,u,node; species=2,value=0.0)
	end
	if data[1]=="nm"
		boundary_neumann!(y,u,node; species=1,value=data[2])
		boundary_neumann!(y,u,node; species=2,value=data[3])
	else
		boundary_dirichlet!(y,u,node; species=1,value=data[2])
		boundary_dirichlet!(y,u,node; species=2,value=data[3])
	end
end

# ╔═╡ 6d54cd64-9913-46d1-813d-0200976dae8b
function bc_3d(y,u,node,data)
	
	if node[1]==0 && node[2]==0 && node[3]==0
		boundary_dirichlet!(y,u,node; species=2,value=0.0)
	end
	if data[1]=="nm"
		boundary_neumann!(y,u,node; species=1,value=data[2])
		boundary_neumann!(y,u,node; species=2,value=data[3])
	else
		boundary_dirichlet!(y,u,node; species=1,value=data[2])
		boundary_dirichlet!(y,u,node; species=2,value=data[3])
	end
end

# ╔═╡ 3574deac-d35b-47f5-910e-2e41c699d7f9
md"""
### Solution Strategy

The nonlinear system of equations are solved using a Newton iteration scheme. During the update step, we utilize BICGstab iteration scheme with an LU or ILU preconditioner (LU factorization is recomputed every n steps, which is an additional hyperparameter). In an attempt to speed up computation, we consider damping, which scales the update vector down by some factor and increases in every iteration step until we reach 1.0.
"""

# ╔═╡ d9b87f2e-590e-4731-b121-6229f210294f
md"""
And now the heart of this notebook: the Bidomain Model function.
"""

# ╔═╡ 59ee920b-22e3-457d-ac44-2cb9fe4c46d2
function bidomain(;
		dim=1,
		tend=50,
		gridlength=70,
		tstep=1.0e-4,
		damp=0.5,
		damp_grow=1.8,
		h_1d=0.007,
		data=["nm",0,0],
		h_2d=0.233,
		h_3d=1.4,
		lureuse=3,
		precon=ExtendableSparse.LUFactorization(),
		verbose=false) 
### check for dimension to choose 1D, 2D or 3D grid, diffusion and boundary conditions.
	if dim==1
		X=0:h_1d:gridlength
		grid=simplexgrid(X)
		bc=bc_1d
		diffu=diffusion
	elseif dim==2
		X=0:h_2d:gridlength
		grid=simplexgrid(X,X)
		bc=bc_2d
		diffu=diffusion_2d
	else
		X=0:h_3d:gridlength
		grid=simplexgrid(X,X,X)
		bc=bc_3d
		diffu=diffusion_3d
	end
	

### creation of the Voronoi System and the initial values matrix.
	system=VoronoiFVM.System(grid,
			flux=diffu,
			storage=storage,
	        reaction=reaction,
			bcondition=bc,
            species=[1,2,3],
			log=true,
			data=data)
	inival=unknowns(system)
	coord=grid[Coordinates]

### filling of the initial values matrix depending on choosen dimension
	if dim==1
		for i=1:size(inival,2)
			if coord[1,i]<=gridlength/20
				inival[1,i]=2.0
			else
				inival[1,i]=((10^(1/2)-3)^(2/3)-1)/((10)^(1/2)-3)^(1/3)
			end
			inival[3,i]=1/γ*(β+((10^(1/2)-3)^(2/3)-1)/((10)^(1/2)-3)^(1/3))
			inival[2,i]=0.0	
		end
	elseif dim==2
		for i=1:size(inival,2)
			
			if coord[1,i]<=gridlength/20
				inival[1,i]=2.0
			else
				inival[1,i]=((10^(1/2)-3)^(2/3)-1)/((10)^(1/2)-3)^(1/3)
			end
### create recovery zone for v 
			if 31<=coord[1,i]<=39 && coord[2,i]<=35
				inival[3,i]=2.0
			else
				inival[3,i]=1/γ*(β+((10^(1/2)-3)^(2/3)-1)/((10)^(1/2)-3)^(1/3))
			end
			inival[2,i]=0.0
				
		end
	elseif dim==3
		for i=1:size(inival,2)
			
			if coord[1,i]<=gridlength/20
				inival[1,i]=2.0
			else
				inival[1,i]=((10^(1/2)-3)^(2/3)-1)/((10)^(1/2)-3)^(1/3)
			end
			if 31<=coord[1,i]<=39 && coord[2,i]<=35
				inival[3,i]=2.0
			else
				inival[3,i]=1/γ*(β+((10^(1/2)-3)^(2/3)-1)/((10)^(1/2)-3)^(1/3))
			end
			inival[2,i]=0.0
				
		end	
	end

### initializing SolverControl and solving the VoronoiFVM system	
	control=VoronoiFVM.SolverControl()
	control.Δt_min=0.01*tstep
	control.Δt=tstep
	control.max_lureuse=lureuse
	control.verbose=verbose
	control.Δt_max=0.01*tend
	control.Δu_opt=2
	control.factorization=precon
	control.max_iterations=1000
	#control.handle_exceptions=true
	tsol=VoronoiFVM.solve(system,inival=inival,times=[0,tend],control=control,log=true,damp=damp,damp_grow=damp_grow)
	#problem = ODEProblem(system,inival,(0,tend))
	#odesol = DifferentialEquations.solve(problem,ImplicitEuler())
	#tsol=reshape(odesol,system)
	return tsol,grid,tend,dim,system
end

# ╔═╡ b08a77ba-96c9-49ee-b5f7-c152d6cba678
md"""
### 1D Bidomain Model
"""

# ╔═╡ 6b78bd4e-bd25-4f37-b1b5-14784cc51ad7

tsol,grid,tend,dim,system=bidomain(dim=1,
		tend=50,
		gridlength=70,
		tstep=1.0e-3,
		damp=0.5,
		damp_grow=1.8,
		data=["nm",0,0],
		lureuse=2,
		precon=ILUTPreconditioner(),
		h_1d=0.07)



# ╔═╡ de485383-9310-4cc4-aaf7-a481ee373d5a
gridplot(grid,resolution=(600,400),linewidth=0.5,legend=:lt)

# ╔═╡ dcfc5fb0-9358-4958-977c-648afb8fd2a0
# ╠═╡ disabled = true
#=╠═╡
begin
	#bam=[ILUTPreconditioner(),ExtendableSparse.LUFactorization()]
	times=zeros((10, 2))
	for i=1:10
		for j=1:2
			if j==1
				k=ILUTPreconditioner()
			else
				k=ExtendableSparse.LUFactorization()
			end
			times[i,j]=@belapsed bidomain(dim=1,
			tend=50,
			gridlength=70,
			tstep=1.0e-3,
			damp=$i/10,
			damp_grow=1.8,
			data=["nm",0,0],
			lureuse=3,
			precon=$k,
			h_1d=0.07)
			
		end
	end
end
  ╠═╡ =#

# ╔═╡ feb45795-4f53-4d96-8b3f-3f64527c4ac3
#=╠═╡
times
  ╠═╡ =#

# ╔═╡ b4aa3985-be2e-472f-94d8-0053375400f3
md"""
Time t=$(@bind t Slider(0:tend/1000:tend,show_value=true,default=11))
"""

# ╔═╡ 9a378aed-bf36-46dc-8499-2405858394dc
vis=GridVisualizer(layout=(1,1);size=(700,350),dim=dim,xlabel="Domain Ω",ylabel="Electrical Potential",legend=:lt)

# ╔═╡ d41a20c6-dac0-47b3-97c0-63a4bffba69c
sol=tsol(t)

# ╔═╡ 8193ce62-d530-49cb-8b13-6711631e7268
scalarplot!(vis,grid,sol[1,:],limits=(-2,2),show=true,label="u",linestyle=:line,color=:blue)

# ╔═╡ fe8e76b7-8cfc-4de9-af3d-24cdc81fa5ab
scalarplot!(vis,grid,sol[2,:],limits=(-2,2),show=true,clear=false,color=:green, label="u_e",linestyle=:dash)

# ╔═╡ e0bce0cf-e991-4c35-b77f-3d891e09b00e
scalarplot!(vis,grid,sol[3,:],limits=(-2,2),show=true,clear=false,color=:red,label="v",linestyle=:dot)

# ╔═╡ 8081c60a-b67c-4346-8094-6746e9b12e76
tsol;history_summary(system)

# ╔═╡ f13f7688-cf00-4c75-924d-498dda77a8b9
md"""
#### Damping and MaxLUreuse benchmarks
"""

# ╔═╡ b516bd42-60bc-4043-b449-4a72e0b1f6db
begin
	df=DataFrame(CSV.File("damp_test.csv"))	
	df[:, [:damp, :ILUT, :LU]]
end

# ╔═╡ d9a60226-1c49-403e-95ad-68fda753e90c
begin
	df2=DataFrame(CSV.File("MaxLU.csv"))	
	df2[:, [:Maxlureuse, :ILUT, :LU]]
end

# ╔═╡ 714a908b-b0d2-4b7e-b99a-2b38c2feace0
md"""
### 2D Bidomain Model
"""

# ╔═╡ 727dd44f-2145-47de-a7e0-024704b4f04e
tsol_2d,grid_2d,tend_2d,dim_2d,system_2d=bidomain(dim=2,
		tend=50,
		gridlength=70,
		tstep=1.0e-2,
		damp=0.5,
		damp_grow=1.8,
		lureuse=3,
		precon=ILUTPreconditioner(),
		h_2d=0.233)
# tend=50, tstep=1.0e-3,h2d=0.233,lureuse: 3, precon: LUfacto(), 3618 s
# tend=50, tstep=1.0e-2,h2d=0.233,lureuse: 3, precon: LUfacto(), 3506 s
# tend=50, tstep=1.0e-2,h2d=0.233,lureuse: 3, precon: ILUTPreconditioner(), 9352 s
# tend=50, tstep=1.0e-2,h2d=0.233,lureuse: 3, precon: LUfacto(), 1009 s
# tend=50, tstep=1.0e-2,h2d=0.233,lureuse: 3, precon: LUfacto(), 855 s /systemrestart
# tend=50, tstep=1.0e-3,h2d=0.466,lureuse: 3, precon: ILUTPreconditioner(), 1489 s

# ╔═╡ 78134d85-5b97-451a-9d90-b3d55e42ea92
gridplot(grid_2d,resolution=(600,400),linewidth=0.5,legend=:lt)

# ╔═╡ f98c536d-4f9d-4b52-8bbd-8c1b38e497a8
md"""
Time t 2D=$(@bind t_2d Slider(0:tend_2d/1000:tend_2d,show_value=true))
"""

# ╔═╡ 7fe94a1c-602b-431e-95ff-b7cbb6bac814
md"""
Species:
"""

# ╔═╡ 0770bc4e-af0d-4e72-8d2f-9d700d3728b6
@bind species Radio(["u","u_e","v"],default="u")


# ╔═╡ 1588044c-319a-4da8-afca-293562496c0f

if species=="u"
	spec=1
elseif species=="u_e"
	spec=3
else
	spec=2
end


# ╔═╡ 57aa2c61-d8d6-4237-8e2e-94426b3c64d5
sol_2d=tsol_2d(t_2d)

# ╔═╡ 9033201c-4bc2-4fc8-a3ee-2829f2cde5fa
vis_2d=GridVisualizer(layout=(1,1);size=(700,350),dim=dim_2d,legend=:lt)

# ╔═╡ e79a02cd-7fdc-41f5-a5fe-1e5fa368650d
scalarplot!(vis_2d,grid_2d,sol_2d[spec,:],limits=(-2,2),show=true,label="2D Grid",linestyle=:line,color=:blue)

# ╔═╡ 964beb31-1fb9-4a21-81db-7f8ad566b430
tsol_2d;history_summary(system_2d)

# ╔═╡ c875f5cc-119d-4b84-89c8-8c10912889b1
md"""
### 3D Bidomain Model
"""

# ╔═╡ f309d847-8b94-4ead-a02d-ae155ff5ec4f
md"""
We implemented a 3D case, but the results were computationally costly and inconclusive. We tried offshoring the computation to a digital ocean server, but it ran out of memory. The tensor used was proportional to the values suggested in [2].
"""

# ╔═╡ 4960ae3c-68d7-4564-a533-23fccefac619
# ╠═╡ disabled = true
#=╠═╡
PlutoUI.with_terminal() do
  tsol_3d,grid_3d,tend_3d,dim_3d,system_3d=bidomain(dim=3,
		tend=20,
		gridlength=70,
		tstep=1.0e-2,
		damp=0.5,
		damp_grow=1.8,
		data=["nm",0,0],
		h_3d=1.4,
		verbose=true)
end

  ╠═╡ =#

# ╔═╡ c32c9069-1bc3-4d7c-8800-ab09c172c2a8
# ╠═╡ disabled = true
#=╠═╡
begin
	grid_3d=load("muiui.jld2", "grid_3d")
	tend_3d=load("muiui.jld2", "tend_3d")
	dim_3d=load("muiui.jld2", "dim_3d")
	tsol_3d=load("muiui.jld2", "tsol_3d")
end
  ╠═╡ =#

# ╔═╡ 2dc84625-ce73-41a6-875f-ed89a606aae7
# ╠═╡ disabled = true
#=╠═╡
gridplot(grid_3d,resolution=(600,400),linewidth=0.5,legend=:lt)
  ╠═╡ =#

# ╔═╡ 14ed1150-d4e7-4faf-93c3-c0e23682106e
# ╠═╡ disabled = true
#=╠═╡
md"""
t 3D=$(@bind t_3d Slider(0:tend_3d/1000:tend_3d,show_value=true))
"""
  ╠═╡ =#

# ╔═╡ 7db43261-ce19-4666-a8f1-3ccfbb69f610
# ╠═╡ disabled = true
#=╠═╡
vis_3d=GridVisualizer(layout=(1,1);size=(700,350),dim=dim_3d,legend=:lt)
  ╠═╡ =#

# ╔═╡ b47c3692-d67c-4161-8073-993be1850e5f
# ╠═╡ disabled = true
#=╠═╡
sol_3d=tsol_3d(t_3d)
  ╠═╡ =#

# ╔═╡ e2c472ca-b117-4496-bfb2-8f229931b079
# ╠═╡ disabled = true
#=╠═╡
scalarplot!(vis_3d,grid_3d,sol_3d[spec,:],limits=(-2,2),show=true,label="3D Grid",color=:blue)
  ╠═╡ =#

# ╔═╡ d3a24181-6c65-4b47-b09d-95cdccf8067f
#jldsave("3d_test.jld2"; tsol_3d,grid_3d,tend_3d,dim_3d)

# ╔═╡ a86e31e1-b8ca-4219-ab15-b679751c9c4f
md"""
## Conclusion
We have demonstrated how to model the bidomain model using the VoronoiFVM package for solving PDEs, and presented solution in 1D and 2D cases, and discussed limitations of this approach for the 3D case. Using first space, then time descretization, we calculate properties of excitation waves traversing cardiac tissues. We relied heavily on [1] for initial conditions, constants, and formulation of equations to propogate this wave over our domain.

We compared 2 methods of incorporating the conduction tensor into our 2D solution, which had little effect on the runtime. We further benchmarked the LU vs ILU factorization,in respect to the damping parameter and number of LU reuses per Newton iteration step, which resulted in favorable outcome for ILU.
"""

# ╔═╡ f12b3187-2ec3-444f-842c-cbf7d2a85041
md"""
## References
"""

# ╔═╡ f09ba0f2-4d9b-492e-98cb-1c050797d792
md"""
[1]Ethier, Marc & Bourgault, Yves. (2008). Semi-Implicit Time-Discretization Schemes for the Bidomain Model. SIAM J. Numerical Analysis. 46. 2443-2468. 10.1137/070680503. 

[2]Darren A. Hooks, Karl A. Tomlinson, Scott G. Marsden, Ian J. LeGrice, Bruce H.
Smaill, Andrew J. Pullan, and Peter J. Hunter, Cardiac microstructure: Implications for electrical propagation and defibrillation in the heart, Circulation Research, 91 (2002), pp. 331–338.
"""

# ╔═╡ 1de1f689-4de9-4d49-8cd3-aeaf4be5ccea
html"<hr>"

# ╔═╡ 5da7074e-19bd-4cc6-aa30-3bf597f4a631
begin
    highlight(mdstring,color)= htl"""<blockquote style="padding: 10px; background-color: $(color);">$(mdstring)</blockquote>"""
	
	macro important_str(s)	:(highlight(Markdown.parse($s),"#ffcccc")) end
	macro definition_str(s)	:(highlight(Markdown.parse($s),"#ccccff")) end
	macro statement_str(s)	:(highlight(Markdown.parse($s),"#ccffcc")) end
		
		
    html"""
    <style>
     h1{background-color:#dddddd;  padding: 10px;}
     h2{background-color:#e7e7e7;  padding: 10px;}
     h3{background-color:#eeeeee;  padding: 10px;}
     h4{background-color:#f7f7f7;  padding: 10px;}
    </style>
"""
end


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AlgebraicMultigrid = "2169fc97-5a83-5252-b627-83903c6c433c"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
ExtendableGrids = "cfc395e8-590f-11e8-1f13-43a2532b2fa8"
ExtendableSparse = "95c220a8-a1cf-11e9-0c77-dbfce5f500b3"
GridVisualize = "5eed8a63-0fb0-45eb-886d-8d5a387d12b8"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
IncompleteLU = "40713840-3770-5561-ab4c-a76e7d0d7895"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PlutoVista = "646e1f28-b900-46d7-9d87-d554eb38a413"
Sundials = "c3572dad-4567-51f8-b174-8c6c989267f4"
VoronoiFVM = "82b139dc-5afc-11e9-35da-9b9bdfd336f3"

[compat]
AlgebraicMultigrid = "~0.5.1"
BenchmarkTools = "~1.3.1"
CSV = "~0.10.4"
DataFrames = "~1.3.4"
DifferentialEquations = "~7.1.0"
ExtendableGrids = "~0.9.5"
ExtendableSparse = "~0.6.7"
GridVisualize = "~0.5.1"
HypertextLiteral = "~0.9.3"
IncompleteLU = "~0.2.0"
JLD2 = "~0.4.22"
JSON3 = "~1.9.4"
PlutoUI = "~0.7.38"
PlutoVista = "~0.8.13"
Sundials = "~4.9.3"
VoronoiFVM = "~0.16.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.AbstractAlgebra]]
deps = ["GroupsCore", "InteractiveUtils", "LinearAlgebra", "MacroTools", "Markdown", "Random", "RandomExtensions", "SparseArrays", "Test"]
git-tree-sha1 = "dd2f52bc149ff35158827471453e2e4f1a2685a6"
uuid = "c3fe647b-3220-5bb0-a1ea-a7954cac585d"
version = "0.26.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.AlgebraicMultigrid]]
deps = ["CommonSolve", "LinearAlgebra", "Printf", "Reexport", "SparseArrays"]
git-tree-sha1 = "796eedcb42226861a51d92d28ee82d4985ee860b"
uuid = "2169fc97-5a83-5252-b627-83903c6c433c"
version = "0.5.1"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "81f0cb60dc994ca17f68d9fb7c942a5ae70d9ee4"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "5.0.8"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c23473c60476e62579c077534b9643ec400f792b"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "0.8.6"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AutoHashEquals]]
git-tree-sha1 = "45bb6705d93be619b81451bb2006b7ee5d4e4453"
uuid = "15f4f7f2-30c1-5605-9d31-71845cf9641f"
version = "0.2.0"

[[deps.BandedMatrices]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "019aa88766e2493c59cbd0a9955e1bac683ffbcd"
uuid = "aae01518-5342-5314-be14-df237901396f"
version = "0.16.13"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "b15a6bc52594f5e4a3b825858d1089618871bf9d"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.36"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

[[deps.Bijections]]
git-tree-sha1 = "705e7822597b432ebe152baa844b49f8026df090"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.1.3"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "28bbdbf0354959db89358d1d79d421ff31ef0b5e"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.3"

[[deps.BoundaryValueDiffEq]]
deps = ["BandedMatrices", "DiffEqBase", "FiniteDiff", "ForwardDiff", "LinearAlgebra", "NLsolve", "Reexport", "SparseArrays"]
git-tree-sha1 = "fe34902ac0c3a35d016617ab7032742865756d7d"
uuid = "764a87c0-6b3e-53db-9096-fe964310641d"
version = "2.7.1"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "Static"]
git-tree-sha1 = "0eaf4aedad5ccc3e39481db55d72973f856dc564"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.1.22"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "873fb188a4b9d76549b81465b1f75c82aaf59238"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.4"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.CloseOpenIntervals]]
deps = ["ArrayInterface", "Static"]
git-tree-sha1 = "f576084239e6bdf801007c80e27e2cc2cd963fe0"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.6"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "7297381ccb5df764549818d9a7d57e45f1057d30"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.18.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "a985dc37e357a3b22b260a5def99f3530fb415d3"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.2"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "3f1f500312161f1ae067abe07d13b40f78f32e07"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.8"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "b153278a25dd42c65abbf4e62344f9d22e59191b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.43.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.CompositeTypes]]
git-tree-sha1 = "d5b014b216dc891e81fea299638e4c10c657b582"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.2"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DEDataArrays]]
deps = ["ArrayInterface", "DocStringExtensions", "LinearAlgebra", "RecursiveArrayTools", "SciMLBase", "StaticArrays"]
git-tree-sha1 = "5e5f8f363c8c9a2415ef9185c4e0ff6966c87d52"
uuid = "754358af-613d-5f8d-9788-280bf1605d4c"
version = "0.2.2"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "daa21eb85147f72e41f6352a57fccea377e310a9"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "cc1a8e22627f33c789ab60b36a9132ac050bbf75"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.12"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelayDiffEq]]
deps = ["ArrayInterface", "DataStructures", "DiffEqBase", "LinearAlgebra", "Logging", "NonlinearSolve", "OrdinaryDiffEq", "Printf", "RecursiveArrayTools", "Reexport", "UnPack"]
git-tree-sha1 = "8b5f26fba11e8fa570a1bcb752e2a4eed2a92ddd"
uuid = "bcd4f6db-9728-5f36-b5f7-82caef46ccdb"
version = "5.35.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffEqBase]]
deps = ["ArrayInterface", "ChainRulesCore", "DEDataArrays", "DataStructures", "Distributions", "DocStringExtensions", "FastBroadcast", "ForwardDiff", "FunctionWrappers", "IterativeSolvers", "LabelledArrays", "LinearAlgebra", "Logging", "MuladdMacro", "NonlinearSolve", "Parameters", "PreallocationTools", "Printf", "RecursiveArrayTools", "RecursiveFactorization", "Reexport", "Requires", "SciMLBase", "Setfield", "SparseArrays", "StaticArrays", "Statistics", "SuiteSparse", "ZygoteRules"]
git-tree-sha1 = "2daced17b3cb7e67fc67656556c8ba0f88b2a040"
uuid = "2b5f629d-d688-5b77-993f-72d75c75574e"
version = "6.86.0"

[[deps.DiffEqCallbacks]]
deps = ["DataStructures", "DiffEqBase", "ForwardDiff", "LinearAlgebra", "NLsolve", "OrdinaryDiffEq", "Parameters", "RecipesBase", "RecursiveArrayTools", "SciMLBase", "StaticArrays"]
git-tree-sha1 = "c4b99e3a199e293e7290eea94ba89364d47ee557"
uuid = "459566f4-90b8-5000-8ac3-15dfb0a30def"
version = "2.22.0"

[[deps.DiffEqJump]]
deps = ["ArrayInterface", "Compat", "DataStructures", "DiffEqBase", "DocStringExtensions", "FunctionWrappers", "Graphs", "LinearAlgebra", "Markdown", "PoissonRandom", "Random", "RandomNumbers", "RecursiveArrayTools", "Reexport", "StaticArrays", "TreeViews", "UnPack"]
git-tree-sha1 = "26d88f58260cb61f9532c2f7283bc6c6519f954d"
uuid = "c894b116-72e5-5b58-be3c-e6d8d4ac2b12"
version = "8.4.2"

[[deps.DiffEqNoiseProcess]]
deps = ["DiffEqBase", "Distributions", "GPUArrays", "LinearAlgebra", "Optim", "PoissonRandom", "QuadGK", "Random", "Random123", "RandomNumbers", "RecipesBase", "RecursiveArrayTools", "ResettableStacks", "SciMLBase", "StaticArrays", "Statistics"]
git-tree-sha1 = "915286127c88b9306e29229cde687d46196f4060"
uuid = "77a26b50-5914-5dd7-bc55-306e6241c503"
version = "5.11.0"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "28d605d9a0ac17118fe2c5e9ce0fbb76c3ceb120"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.0"

[[deps.DifferentialEquations]]
deps = ["BoundaryValueDiffEq", "DelayDiffEq", "DiffEqBase", "DiffEqCallbacks", "DiffEqJump", "DiffEqNoiseProcess", "LinearAlgebra", "LinearSolve", "OrdinaryDiffEq", "Random", "RecursiveArrayTools", "Reexport", "SteadyStateDiffEq", "StochasticDiffEq", "Sundials"]
git-tree-sha1 = "3f3db9365fedd5fdbecebc3cce86dfdfe5c43c50"
uuid = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
version = "7.1.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "8a6b49396a4058771c5c072239b2e0a76e2e898c"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.58"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "StaticArrays", "Statistics"]
git-tree-sha1 = "5f5f0b750ac576bcf2ab1d7782959894b304923e"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.5.9"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.DynamicPolynomials]]
deps = ["DataStructures", "Future", "LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Pkg", "Reexport", "Test"]
git-tree-sha1 = "d0fa82f39c2a5cdb3ee385ad52bc05c42cb4b9f0"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.4.5"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.ElasticArrays]]
deps = ["Adapt"]
git-tree-sha1 = "a0fcc1bb3c9ceaf07e1d0529c9806ce94be6adf9"
uuid = "fdbdab4c-e67f-52f5-8c3f-e7b388dad3d4"
version = "1.2.9"

[[deps.EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "010c3f9692344e56d05793311dfe554b0d351d79"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.5.1"

[[deps.ExponentialUtilities]]
deps = ["ArrayInterface", "GPUArrays", "GenericSchur", "LinearAlgebra", "Printf", "SparseArrays", "libblastrampoline_jll"]
git-tree-sha1 = "8173af6a65279017e564121ce940bb84ca9a35c9"
uuid = "d4d017d3-3776-5f7e-afef-a10c40355c18"
version = "1.16.0"

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[deps.ExtendableGrids]]
deps = ["AbstractTrees", "Dates", "DocStringExtensions", "ElasticArrays", "InteractiveUtils", "LinearAlgebra", "Printf", "Random", "SparseArrays", "StaticArrays", "Test", "WriteVTK"]
git-tree-sha1 = "339339357704f5e5ee18c3e0d8e1c004667e66ee"
uuid = "cfc395e8-590f-11e8-1f13-43a2532b2fa8"
version = "0.9.6"

[[deps.ExtendableSparse]]
deps = ["DocStringExtensions", "LinearAlgebra", "Printf", "Requires", "SparseArrays", "SuiteSparse", "Test"]
git-tree-sha1 = "eb3393e4de326349a4b5bccd9b17ed1029a2d0ca"
uuid = "95c220a8-a1cf-11e9-0c77-dbfce5f500b3"
version = "0.6.7"

[[deps.FastBroadcast]]
deps = ["LinearAlgebra", "Polyester", "Static"]
git-tree-sha1 = "b6bf57ec7a3f294c97ae46124705a9e6b906a209"
uuid = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
version = "0.1.15"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "9267e5f50b0e12fdfd5a2455534345c4cf2c7f7a"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.14.0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "129b104185df66e408edd6625d480b7f9e9823a0"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.18"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "51c8f36c81badaa0e9ec405dcbabaf345ed18c84"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.11.1"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2f18915445b248731ec5db4e4a17e451020bf21e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.30"

[[deps.FunctionWrappers]]
git-tree-sha1 = "241552bc2209f0fa068b6415b1942cc0aa486bcc"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GPUArrays]]
deps = ["Adapt", "LLVM", "LinearAlgebra", "Printf", "Random", "Serialization", "Statistics"]
git-tree-sha1 = "c783e8883028bf26fb05ed4022c450ef44edd875"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.3.2"

[[deps.GenericSchur]]
deps = ["LinearAlgebra", "Printf"]
git-tree-sha1 = "fb69b2a645fa69ba5f474af09221b9308b160ce6"
uuid = "c145ed77-6b09-5dd9-b285-bf645a82121e"
version = "0.5.3"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "57c021de207e234108a6f1454003120a1bf350c4"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.6.0"

[[deps.GridVisualize]]
deps = ["ColorSchemes", "Colors", "DocStringExtensions", "ElasticArrays", "ExtendableGrids", "GeometryBasics", "HypertextLiteral", "LinearAlgebra", "Observables", "OrderedCollections", "PkgVersion", "Printf", "StaticArrays"]
git-tree-sha1 = "5d845bccf5d690879f4f5f01c7112e428b1fa543"
uuid = "5eed8a63-0fb0-45eb-886d-8d5a387d12b8"
version = "0.5.1"

[[deps.Groebner]]
deps = ["AbstractAlgebra", "Combinatorics", "Logging", "MultivariatePolynomials", "Primes", "Random"]
git-tree-sha1 = "2b40a33e4a6ada7477312c560b6d9fd53f20dee1"
uuid = "0b43b601-686d-58a3-8a1c-6623616c7cd4"
version = "0.2.5"

[[deps.GroupsCore]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9e1a5e9f3b81ad6a5c613d181664a0efc6fe6dd7"
uuid = "d5909c97-4eac-4ecc-a3dc-fdd0858a4120"
version = "0.4.0"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "18be5268cf415b5e27f34980ed25a7d34261aa83"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.7"

[[deps.Hwloc]]
deps = ["Hwloc_jll"]
git-tree-sha1 = "92d99146066c5c6888d5a3abc871e6a214388b91"
uuid = "0e44f5e4-bd66-52a0-8798-143a42290a1d"
version = "2.0.0"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "303d70c961317c4c20fafaf5dbe0e6d610c38542"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.7.1+0"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "cb7099a0109939f16a4d3b572ba8396b1f6c7c31"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.10"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.IncompleteLU]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "a22b92ffedeb499383720dfedcd473deb9608b62"
uuid = "40713840-3770-5561-ab4c-a76e7d0d7895"
version = "0.2.0"

[[deps.Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "61feba885fac3a407465726d0c330b3055df897f"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.2"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "f366daebdfb079fd1fe4e3d560f99a0c892e15bc"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "bcf640979ee55b652f3b01650444eb7bbe3ea837"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.4"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "336cc738f03e069ef2cac55a104eb823455dca75"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.4"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1169632f425f79429f245113b775a0e3d121457c"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "81b9477b49402b47fbe7f7ae0b252077f53e4a08"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.22"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "StructTypes", "UUIDs"]
git-tree-sha1 = "fd6f0cae36f42525567108a42c1c674af2ac620d"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.9.5"

[[deps.KLU]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse_jll"]
git-tree-sha1 = "cae5e3dfd89b209e01bcd65b3a25e74462c67ee0"
uuid = "ef3ab10e-7fda-4108-b977-705223b18434"
version = "0.3.0"

[[deps.Krylov]]
deps = ["LinearAlgebra", "Printf", "SparseArrays"]
git-tree-sha1 = "13b16b00144816211cbf92823ded6042490eb009"
uuid = "ba0b0d4f-ebba-5204-a429-3ac8c609bfb7"
version = "0.8.1"

[[deps.KrylovKit]]
deps = ["LinearAlgebra", "Printf"]
git-tree-sha1 = "49b0c1dd5c292870577b8f58c51072bd558febb9"
uuid = "0b1a1467-8014-51b9-945f-bf0ae24f4b77"
version = "0.5.4"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "c8d47589611803a0f3b4813d9e267cd4e3dbcefb"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.11.1"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "771bfe376249626d3ca12bcd58ba243d3f961576"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.16+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LabelledArrays]]
deps = ["ArrayInterface", "ChainRulesCore", "LinearAlgebra", "MacroTools", "StaticArrays"]
git-tree-sha1 = "1cccf6d366e51fbaf80303158d49bb2171acfeee"
uuid = "2ee39098-c373-598a-b85f-a56591580800"
version = "1.9.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "46a39b9c58749eefb5f2dc1178cb8fab5332b1ab"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.15"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static"]
git-tree-sha1 = "b651f573812d6c36c22c944dd66ef3ab2283dfa1"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.6"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LevyArea]]
deps = ["LinearAlgebra", "Random", "SpecialFunctions"]
git-tree-sha1 = "56513a09b8e0ae6485f34401ea9e2f31357958ec"
uuid = "2d8b4e74-eb68-11e8-0fb9-d5eb67b50637"
version = "1.0.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.LightXML]]
deps = ["Libdl", "XML2_jll"]
git-tree-sha1 = "e129d9391168c677cd4800f5c0abb1ed8cb3794f"
uuid = "9c8b4983-aa76-5018-a973-4c85ecc9e179"
version = "0.9.0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearSolve]]
deps = ["ArrayInterface", "DocStringExtensions", "GPUArrays", "IterativeSolvers", "KLU", "Krylov", "KrylovKit", "LinearAlgebra", "RecursiveFactorization", "Reexport", "SciMLBase", "Setfield", "SparseArrays", "SuiteSparse", "UnPack"]
git-tree-sha1 = "46916e2f4b244592a115d4dd742ccad54571d858"
uuid = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
version = "1.16.3"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "ChainRulesCore", "CloseOpenIntervals", "DocStringExtensions", "ForwardDiff", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "SIMDDualNumbers", "SLEEFPirates", "SpecialFunctions", "Static", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "4392c19f0203df81512b6790a0a67446650bdce0"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.110"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Metatheory]]
deps = ["AutoHashEquals", "DataStructures", "Dates", "DocStringExtensions", "Parameters", "Reexport", "TermInterface", "ThreadsX", "TimerOutputs"]
git-tree-sha1 = "0886d229caaa09e9f56bcf1991470bd49758a69f"
uuid = "e9d8d322-4543-424a-9be4-0cc815abe26c"
version = "1.3.3"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "6bb7786e4f24d44b4e29df03c69add1b63d88f01"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MuladdMacro]]
git-tree-sha1 = "c6190f9a7fc5d9d5915ab29f2134421b12d24a68"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.2"

[[deps.MultivariatePolynomials]]
deps = ["ChainRulesCore", "DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "393fc4d82a73c6fe0e2963dd7c882b09257be537"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.4.6"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "4050cd02756970414dab13b55d55ae1826b19008"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.0.2"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50310f934e55e5ca3912fb941dec199b49ca9b68"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.2"

[[deps.NLsolve]]
deps = ["Distances", "LineSearches", "LinearAlgebra", "NLSolversBase", "Printf", "Reexport"]
git-tree-sha1 = "019f12e9a1a7880459d0173c182e6a99365d7ac1"
uuid = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
version = "4.5.1"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.NonlinearSolve]]
deps = ["ArrayInterface", "FiniteDiff", "ForwardDiff", "IterativeSolvers", "LinearAlgebra", "RecursiveArrayTools", "RecursiveFactorization", "Reexport", "SciMLBase", "Setfield", "StaticArrays", "UnPack"]
git-tree-sha1 = "aeebff6a2a23506e5029fd2248a26aca98e477b3"
uuid = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
version = "0.3.16"

[[deps.Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "52addd9e91df8a6a5781e5c7640787525fd48056"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.11.2"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "7a28efc8e34d5df89fc87343318b0a8add2c4021"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.OrdinaryDiffEq]]
deps = ["Adapt", "ArrayInterface", "DataStructures", "DiffEqBase", "DocStringExtensions", "ExponentialUtilities", "FastClosures", "FiniteDiff", "ForwardDiff", "LinearAlgebra", "LinearSolve", "Logging", "LoopVectorization", "MacroTools", "MuladdMacro", "NLsolve", "NonlinearSolve", "Polyester", "PreallocationTools", "RecursiveArrayTools", "Reexport", "SciMLBase", "SparseArrays", "SparseDiffTools", "StaticArrays", "UnPack"]
git-tree-sha1 = "8031a288c9b418664a3dfbac36e464a3f61ace73"
uuid = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
version = "6.10.0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "027185efff6be268abbaf30cfd53ca9b59e3c857"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.10"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.PlutoVista]]
deps = ["ColorSchemes", "Colors", "DocStringExtensions", "GridVisualize", "HypertextLiteral", "UUIDs"]
git-tree-sha1 = "118d1871e3511131bae2196e238d0054bd9a62b0"
uuid = "646e1f28-b900-46d7-9d87-d554eb38a413"
version = "0.8.13"

[[deps.PoissonRandom]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "44d018211a56626288b5d3f8c6497d28c26dc850"
uuid = "e409e4f3-bfea-5376-8464-e040bb5c01ab"
version = "0.4.0"

[[deps.Polyester]]
deps = ["ArrayInterface", "BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "ManualMemory", "PolyesterWeave", "Requires", "Static", "StrideArraysCore", "ThreadingUtilities"]
git-tree-sha1 = "0578fa5fde97f8cf19aa89f8373d92624314f547"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.6.9"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "7e597df97e46ffb1c8adbaddfa56908a7a20194b"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.1.5"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "ForwardDiff", "LabelledArrays"]
git-tree-sha1 = "6c138c8510111fa47b5d2ed8ada482d97e279bee"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "0.2.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "747f4261ebe38a2bc6abf0850ea8c6d9027ccd07"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "afeacaecf4ed1649555a19cb2cad3c141bbc9474"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.5.0"

[[deps.RandomExtensions]]
deps = ["Random", "SparseArrays"]
git-tree-sha1 = "062986376ce6d394b23d5d90f01d81426113a3c9"
uuid = "fb686558-2515-59ef-acaa-46db3789a887"
version = "0.4.3"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "ChainRulesCore", "DocStringExtensions", "FillArrays", "GPUArrays", "LinearAlgebra", "RecipesBase", "StaticArrays", "Statistics", "ZygoteRules"]
git-tree-sha1 = "6b25d6ba6361ccba58be1cf9ab710e69f6bc96f8"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.27.1"

[[deps.RecursiveFactorization]]
deps = ["LinearAlgebra", "LoopVectorization", "Polyester", "StrideArraysCore", "TriangularSolve"]
git-tree-sha1 = "a9a852c7ebb08e2a40e8c0ab9830a744fa283690"
uuid = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
version = "0.2.10"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Referenceables]]
deps = ["Adapt"]
git-tree-sha1 = "e681d3bfa49cd46c3c161505caddf20f0e62aaa9"
uuid = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"
version = "0.1.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.ResettableStacks]]
deps = ["StaticArrays"]
git-tree-sha1 = "256eeeec186fa7f26f2801732774ccf277f05db9"
uuid = "ae5879a3-cd67-5da8-be7f-38c6eb64a37b"
version = "1.1.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "cdc1e4278e91a6ad530770ebb327f9ed83cf10c4"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.3"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SIMDDualNumbers]]
deps = ["ForwardDiff", "IfElse", "SLEEFPirates", "VectorizationBase"]
git-tree-sha1 = "62c2da6eb66de8bb88081d20528647140d4daa0e"
uuid = "3cdde19b-5bb0-4aaf-8931-af3e248e098b"
version = "0.1.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "ac399b5b163b9140f9c310dfe9e9aaa225617ff6"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.32"

[[deps.SciMLBase]]
deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "RecipesBase", "RecursiveArrayTools", "StaticArrays", "Statistics", "Tables", "TreeViews"]
git-tree-sha1 = "7586a94109dd610b864d10026b5e6a6d481ccaaf"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.32.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "6a2f7d70512d205ca8c7ee31bfa9f142fe74310c"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.12"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "38d88503f695eb0301479bc9b0d4320b378bafe5"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SparseDiffTools]]
deps = ["Adapt", "ArrayInterface", "Compat", "DataStructures", "FiniteDiff", "ForwardDiff", "Graphs", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays", "VertexSafeGraphs"]
git-tree-sha1 = "314a07e191ea4a5ea5a2f9d6b39f03833bde5e08"
uuid = "47a9eef4-7e08-11e9-0b38-333d64bd3804"
version = "1.21.0"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "bc40f042cfcc56230f781d92db71f0e21496dffd"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.5"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "3a2a99b067090deb096edecec1dc291c5b4b31cb"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.6.5"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "cd56bf18ed715e8b09f06ef8c6b781e6cdc49911"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c82aaa13b44ea00134f8c9c89819477bd3986ecd"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.3.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.SteadyStateDiffEq]]
deps = ["DiffEqBase", "DiffEqCallbacks", "LinearAlgebra", "NLsolve", "Reexport", "SciMLBase"]
git-tree-sha1 = "3e057e1f9f12d18cac32011aed9e61eef6c1c0ce"
uuid = "9672c7b4-1e72-59bd-8a11-6ac3964bc41f"
version = "1.6.6"

[[deps.StochasticDiffEq]]
deps = ["Adapt", "ArrayInterface", "DataStructures", "DiffEqBase", "DiffEqJump", "DiffEqNoiseProcess", "DocStringExtensions", "FillArrays", "FiniteDiff", "ForwardDiff", "LevyArea", "LinearAlgebra", "Logging", "MuladdMacro", "NLsolve", "OrdinaryDiffEq", "Random", "RandomNumbers", "RecursiveArrayTools", "Reexport", "SparseArrays", "SparseDiffTools", "StaticArrays", "UnPack"]
git-tree-sha1 = "045f257a43ef3fbc543cd38f64e1188eb07554ce"
uuid = "789caeaf-c7a9-5a7d-9973-96adeb23e2a0"
version = "6.47.1"

[[deps.StrideArraysCore]]
deps = ["ArrayInterface", "CloseOpenIntervals", "IfElse", "LayoutPointers", "ManualMemory", "Requires", "SIMDTypes", "Static", "ThreadingUtilities"]
git-tree-sha1 = "e03eacc0b8c1520e73aa84922ce44a14f024b210"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.3.6"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "e75d82493681dfd884a357952bbd7ab0608e1dc3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.7"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "d24a825a95a6d98c385001212dc9020d609f2d4f"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.8.1"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"

[[deps.Sundials]]
deps = ["CEnum", "DataStructures", "DiffEqBase", "Libdl", "LinearAlgebra", "Logging", "Reexport", "SparseArrays", "Sundials_jll"]
git-tree-sha1 = "6549d3b1b5cf86446949c62616675588159ea2fb"
uuid = "c3572dad-4567-51f8-b174-8c6c989267f4"
version = "4.9.4"

[[deps.Sundials_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg", "SuiteSparse_jll"]
git-tree-sha1 = "04777432d74ec5bc91ca047c9e0e0fd7f81acdb6"
uuid = "fb77eaff-e24c-56d4-86b1-d163f2edb164"
version = "5.2.1+0"

[[deps.SymbolicUtils]]
deps = ["AbstractTrees", "Bijections", "ChainRulesCore", "Combinatorics", "ConstructionBase", "DataStructures", "DocStringExtensions", "DynamicPolynomials", "IfElse", "LabelledArrays", "LinearAlgebra", "Metatheory", "MultivariatePolynomials", "NaNMath", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "TermInterface", "TimerOutputs"]
git-tree-sha1 = "bfa211c9543f8c062143f2a48e5bcbb226fd790b"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "0.19.7"

[[deps.Symbolics]]
deps = ["ArrayInterface", "ConstructionBase", "DataStructures", "DiffRules", "Distributions", "DocStringExtensions", "DomainSets", "Groebner", "IfElse", "Latexify", "Libdl", "LinearAlgebra", "MacroTools", "Metatheory", "NaNMath", "RecipesBase", "Reexport", "Requires", "RuntimeGeneratedFunctions", "SciMLBase", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicUtils", "TermInterface", "TreeViews"]
git-tree-sha1 = "38381b90065c4e444fcdca49b8280ba3571059f8"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "4.5.1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TermInterface]]
git-tree-sha1 = "7aa601f12708243987b88d1b453541a75e3d8c7a"
uuid = "8ea1fca8-c5ef-4a55-8b96-4e9afe9c9a3c"
version = "0.2.3"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "f8629df51cab659d70d2e5618a430b4d3f37f2c3"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.0"

[[deps.ThreadsX]]
deps = ["ArgCheck", "BangBang", "ConstructionBase", "InitialValues", "MicroCollections", "Referenceables", "Setfield", "SplittablesBase", "Transducers"]
git-tree-sha1 = "d223de97c948636a4f34d1f84d92fd7602dc555b"
uuid = "ac1d9e8a-700a-412c-b207-f0111f4b6c0d"
version = "0.1.10"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "7638550aaea1c9a1e86817a231ef0faa9aca79bd"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.19"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "c76399a3bbe6f5a88faa33c8f8a65aa631d95013"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.73"

[[deps.TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[deps.TriangularSolve]]
deps = ["CloseOpenIntervals", "IfElse", "LayoutPointers", "LinearAlgebra", "LoopVectorization", "Polyester", "Static", "VectorizationBase"]
git-tree-sha1 = "b8d08f55b02625770c09615d96927b3a8396925e"
uuid = "d5829a12-d9aa-46ab-831f-fb7c9ab06edf"
version = "0.1.11"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "Hwloc", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static"]
git-tree-sha1 = "b86092766ccbb59aefa7e8c6fa01b10e1934e78c"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.32"

[[deps.VertexSafeGraphs]]
deps = ["Graphs"]
git-tree-sha1 = "8351f8d73d7e880bfc042a8b6922684ebeafb35c"
uuid = "19fa3120-7c27-5ec5-8db8-b0b0aa330d6f"
version = "0.2.0"

[[deps.VoronoiFVM]]
deps = ["DiffResults", "DocStringExtensions", "ExtendableGrids", "ExtendableSparse", "ForwardDiff", "GridVisualize", "IterativeSolvers", "JLD2", "LinearAlgebra", "Parameters", "Printf", "RecursiveArrayTools", "Requires", "SparseArrays", "SparseDiffTools", "StaticArrays", "Statistics", "SuiteSparse", "Symbolics", "Test"]
git-tree-sha1 = "254b5472a9f3ec970a08b24b76cba4bf1d79f3e6"
uuid = "82b139dc-5afc-11e9-35da-9b9bdfd336f3"
version = "0.16.3"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WriteVTK]]
deps = ["Base64", "CodecZlib", "FillArrays", "LightXML", "TranscodingStreams"]
git-tree-sha1 = "bff2f6b5ff1e60d89ae2deba51500ce80014f8f6"
uuid = "64499a7a-5c06-52f2-abe2-ccb03c286192"
version = "1.14.2"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─b86d278f-2317-4247-92e9-50016d43ace5
# ╠═60941eaa-1aea-11eb-1277-97b991548781
# ╟─ab85244a-a833-4a0f-a073-0ec1f5d86cd7
# ╟─db44c771-04ad-4235-a74c-9608995d93fb
# ╟─087a9eac-54ad-4b3e-8479-ddf5b92dca35
# ╟─95a69409-90bd-44bf-b0d9-eb6825e70c02
# ╟─c0162030-2b79-4bf7-976c-f02b65b7f98f
# ╟─0f528d1d-2e1f-4260-928a-5483f23a2169
# ╟─80892b19-8b1a-49f8-a6b2-8e9c19f1342a
# ╟─7f8705c3-a457-4c1c-8581-e1b56e0269c1
# ╟─9e4722e2-1ca3-4758-a397-806a49c0df7a
# ╟─74d89019-bad6-49c3-950b-9ed1742bbba7
# ╟─87f4dbfb-890a-41c6-8b11-3e8d6ba14ee9
# ╟─788c3001-c6a0-4de0-aa3a-cdaa7dcc4c26
# ╠═7d7f31b2-f2e4-47da-ac4a-a67833649bcf
# ╟─7ff84bfd-9245-4f47-ad95-127904a1fa26
# ╠═1e9cadb3-0ee1-46cb-ba9e-39dfcf91f6ae
# ╟─1a474e89-b240-4649-854a-acb10dc8f905
# ╠═22d57dfe-a6b5-48c8-9452-4114a30a85b0
# ╟─c0bd0f63-cb5c-4b1f-8aad-4de419182574
# ╠═910bb2da-67f6-4661-90f6-9ad1e23d7524
# ╠═9f981a30-3505-4d4a-9908-fa9499b0d1cc
# ╠═c0a55467-ce0d-4093-a87a-d88ac402c683
# ╠═1cf82863-6529-4f05-a152-7415b0837041
# ╟─ccbd19e6-4dbc-4f18-b510-64a8ef9b63a3
# ╠═b99bd145-a064-4837-bcf4-f320f4f22426
# ╟─cb4d82c6-e69d-4b06-baf7-d28a0951adce
# ╠═10e2756b-f23c-4f1a-8e2a-7016d46e5d69
# ╠═a8f21213-8866-4fdd-b48d-aec531bd9200
# ╠═6d54cd64-9913-46d1-813d-0200976dae8b
# ╟─3574deac-d35b-47f5-910e-2e41c699d7f9
# ╟─d9b87f2e-590e-4731-b121-6229f210294f
# ╠═59ee920b-22e3-457d-ac44-2cb9fe4c46d2
# ╟─b08a77ba-96c9-49ee-b5f7-c152d6cba678
# ╠═6b78bd4e-bd25-4f37-b1b5-14784cc51ad7
# ╠═de485383-9310-4cc4-aaf7-a481ee373d5a
# ╠═dcfc5fb0-9358-4958-977c-648afb8fd2a0
# ╠═feb45795-4f53-4d96-8b3f-3f64527c4ac3
# ╟─b4aa3985-be2e-472f-94d8-0053375400f3
# ╠═9a378aed-bf36-46dc-8499-2405858394dc
# ╟─d41a20c6-dac0-47b3-97c0-63a4bffba69c
# ╠═8193ce62-d530-49cb-8b13-6711631e7268
# ╠═fe8e76b7-8cfc-4de9-af3d-24cdc81fa5ab
# ╠═e0bce0cf-e991-4c35-b77f-3d891e09b00e
# ╠═8081c60a-b67c-4346-8094-6746e9b12e76
# ╟─f13f7688-cf00-4c75-924d-498dda77a8b9
# ╟─b516bd42-60bc-4043-b449-4a72e0b1f6db
# ╟─d9a60226-1c49-403e-95ad-68fda753e90c
# ╟─714a908b-b0d2-4b7e-b99a-2b38c2feace0
# ╠═727dd44f-2145-47de-a7e0-024704b4f04e
# ╠═78134d85-5b97-451a-9d90-b3d55e42ea92
# ╟─f98c536d-4f9d-4b52-8bbd-8c1b38e497a8
# ╟─7fe94a1c-602b-431e-95ff-b7cbb6bac814
# ╟─0770bc4e-af0d-4e72-8d2f-9d700d3728b6
# ╟─1588044c-319a-4da8-afca-293562496c0f
# ╠═57aa2c61-d8d6-4237-8e2e-94426b3c64d5
# ╠═9033201c-4bc2-4fc8-a3ee-2829f2cde5fa
# ╠═e79a02cd-7fdc-41f5-a5fe-1e5fa368650d
# ╠═964beb31-1fb9-4a21-81db-7f8ad566b430
# ╟─c875f5cc-119d-4b84-89c8-8c10912889b1
# ╟─f309d847-8b94-4ead-a02d-ae155ff5ec4f
# ╠═4960ae3c-68d7-4564-a533-23fccefac619
# ╠═c32c9069-1bc3-4d7c-8800-ab09c172c2a8
# ╠═2dc84625-ce73-41a6-875f-ed89a606aae7
# ╟─14ed1150-d4e7-4faf-93c3-c0e23682106e
# ╠═7db43261-ce19-4666-a8f1-3ccfbb69f610
# ╠═e2c472ca-b117-4496-bfb2-8f229931b079
# ╟─b47c3692-d67c-4161-8073-993be1850e5f
# ╠═d3a24181-6c65-4b47-b09d-95cdccf8067f
# ╟─a86e31e1-b8ca-4219-ab15-b679751c9c4f
# ╟─f12b3187-2ec3-444f-842c-cbf7d2a85041
# ╟─f09ba0f2-4d9b-492e-98cb-1c050797d792
# ╟─1de1f689-4de9-4d49-8cd3-aeaf4be5ccea
# ╟─5da7074e-19bd-4cc6-aa30-3bf597f4a631
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
