### A Pluto.jl notebook ###
# v0.18.4

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
	ENV["LC_NUMERIC"]="C"
	using PlutoUI,PyPlot, Triangulate,SparseArrays, Printf
	using HypertextLiteral
	PyPlot.svg(true);
end;

# ╔═╡ bb74bef3-dab3-46e5-93f4-464ccf5d558c
html"""
<b> Scientific Computing TU Berlin Winter 2021/22  &copy; Jürgen Fuhrmann </b>
<br><b> Notebook 23</b>
"""

# ╔═╡ d1edd6da-55f6-11eb-2411-8f7d467a6ce3
TableOfContents(title="Contents",indent=true,aside=true, depth=4)

# ╔═╡ 9eae02a6-5603-11eb-1384-7f81fe7e4a3d
md"""
# Implementation of the finite volume method

Here, we specifically introduce the Voronoi finite volume method on triangular grids.

We discuss the implementation of the method for the problem


$\begin{align}
  -\nabla\cdot \delta  \vec\nabla u&=f\\
   \delta \partial_n u + \alpha u &= g
\end{align}$

"""

# ╔═╡ e7ff17f0-55f3-11eb-3dbe-31037e82efd9
md"""
## Geometrical data for finite volumes

As seen in the previous lecture, we need to be able to calculate the contributions
to the Voronoi cell data for each triangle.
"""

# ╔═╡ 8eef4836-55f2-11eb-0ae1-a1e54cf4163d
md"""
PA=[$(@bind PA1 Slider(0:0.01:5,default=3,show_value=true)),
    $(@bind PA2 Slider(1.0:0.01:5,default=3,show_value=true))]
"""

# ╔═╡ b8d88140-55ec-11eb-1ce0-b139f9bcc46f
let
		
	PA=[PA1,PA2]
	PC=[5.0,0.0]
	PB=[0,0]

	line(p1,p2;color=:black)=PyPlot.plot([p1[1],p2[1]],[p1[2],p2[2]],"-",color=color)
	text(p,txt;fontsize=15)=PyPlot.text(p[1],p[2],txt,fontsize=fontsize)
	circumcenter(PA,PB,PC)=Triangulate.tricircumcenter!([0.0,0.0],PA,PB,PC)
	edgecenter(PA,PB)=[(PA[1]+PB[1])/2,(PA[2]+PB[2])/2]
	clf()
	ax=PyPlot.axes(aspect=1.0)
	CC=circumcenter(PA,PB,PC)
    line(PA,PB)
    line(PB,PC)
	line(PA,PC)
	line(CC,edgecenter(PA,PB),color=:lightgreen)
	line(CC,edgecenter(PA,PC),color=:lightgreen)
	line(CC,edgecenter(PB,PC),color=:lightgreen)
	
text(PB-[0.4,0], L"$P_B$")
text(PB+[0.7,0.2], L"$\omega_B$")
text(PA+[0,0.2], L"$P_A$")
text(PA-[0.1,0.9], L"$\omega_A$")
text(PC+[-0.9,0.2], L"$\omega_C$")
text(PC+[0.2,0], L"$P_C$")
text(CC+[-0.2,0.3], L"$P_{CC}$")
text(edgecenter(edgecenter(PB,PC),CC)+[0.1,0], L"${s}_a$")
text(edgecenter(edgecenter(PA,PC),CC)+[0,0.2], L"${s}_b$")
text(edgecenter(edgecenter(PA,PB),CC)+[0,0.2], L"${s}_c$")
text(edgecenter(PB,PC)+[0.5,-0.3], L"$a$")
text(edgecenter(PB,PC)+[-0.1,-0.3], L"$P_{BC}$")
text(edgecenter(PA,PC)+[-0.2,0.5], L"$b$")
text(edgecenter(PA,PC)+[0.1,0.1], L"$P_{AC}$")
text(edgecenter(PA,PB)+[0.0,0.5], L"$c$")
text(edgecenter(PA,PB)+[-0.5,0.0], L"$P_{AB}$")
axis("off")

	ax.get_xaxis().set_visible(false)
    ax.get_yaxis().set_visible(false)

	gcf().set_size_inches(5,5)
	gcf()
end

# ╔═╡ 1a7f1e8e-55f4-11eb-3a18-7f34aca5b559
md"""
### Needed data

- Edge lengths $h_{kl}$:
$a=|P_BP_C|, b=|P_AP_C|, c=|P_AP_B|$

- Contributions to lengths of the interfaces between Voronoi cells $|\sigma_{kl}\cap T|$ --  $s_a, s_b, s_c$: length fo lines joining the corresponding edge centers $P_{BC}, P_{AC}, P_{AB}$ with the triangle circumcenter $P_{CC}$. 

- Practically, we need the values of the  ratios $\frac{\sigma_{kl}}{h_{kl}}$:
$e_a=\frac{s_a}a, e_b=\frac{s_b}b, e_c=\frac{s_c}c$

- Triangle contributions to the Voronoi cell areas around the respective triangle nodes $\omega_A=|P_AP_{AB}P_{CC}P_{AC}|, \omega_B=|P_BP_{BC}P_{CC}P_{AB}|, \omega_C=|P_CP_{AC}P_{CC}P_{BC}|$

"""

# ╔═╡ 6209efdc-55f8-11eb-1fa5-49545391945d
md"""
### Calculation steps for the interface contributions
We show the calculation steps for $e_a, \omega_a$, the others can be obtained via corresponding permutations.

1. Semiperimeter: 
$s= \frac{a}{2} + \frac{b}{2} + \frac{c}{2}$

2. Square area (from Heron's formula): 
$16A^2= 16s(s-a)(s-b)(s-c)= \left(- a + b + c\right) \left(a - b + c\right) \left(a + b - c\right) \left(a + b + c\right)$

3. Square circumradius: 
$R^2= \frac{a^{2} b^{2} c^{2}}{\left(- a + b + c\right) \left(a - b + c\right) \left(a + b - c\right) \left(a + b + c\right)} =\frac{a^2b^2c^2}{16A^2}$

4. Square of the Voronoi interface contribution via Pythagoras: 
$s_a^2 = R^2-\left(\frac12 a\right)^2= - \frac{a^{2} \left(a^{2} - b^{2} - c^{2}\right)^{2}}{4 \left(a - b - c\right) \left(a - b + c\right) \left(a + b - c\right) \left(a + b + c\right)}$

5. Square of interface contribution over edge length: 
$e_a^2=\frac{s_a^2}{a^2}= - \frac{\left(a^{2} - b^{2} - c^{2}\right)^{2}}{4 \left(a - b - c\right) \left(a - b + c\right) \left(a + b - c\right) \left(a + b + c\right)} = \frac{(b^2+c^2-a^2)^2}{64A^2}$

6. Interface contribution over edge length:
$e_a=\frac{s_a}{a}=\frac{b^2+c^2-a^2}{8A}$

7. Calculation of the area contributions
$\omega_a=\frac14 cs_c + \frac14 bs_b = \frac14(c^2e_c +b^2e_b)$

"""

# ╔═╡ afc9f670-5604-11eb-23c9-cd64f9f404b8
md"""
- The sign chosen implies a positive value if the angle $\alpha_A  <\frac\pi2$, and a  negative value if it  is obtuse. In the  latter case, this  corresponds to  the negative length  of the line  between edge  midpoint and  circumcenter, which  is exactly  the value which  needs to be  added to  the corresponding amount  from the opposite triangle in order to obtain the measure of the Voronoi face.
- If an edge between two triangles is not locally Delaunay, the summary contribution from the two triangles with respect to this edge will become negative.
"""

# ╔═╡ 1bcbe0aa-55fa-11eb-13ea-6906b6fcae24
md"""
## Steps to  the implementation

We describe a triangular discretization mesh by three arrays:

- `pointlist`:  $2\times n_{points}$ floating point array of node coordinates of the triangulations. `pointlist[:,i]` then contains the coordinates of point `i`.
- `trianglelist`: $3\times n_{ntri}$ integer array describing which three nodes belong to a given triangle. `trianglelist[:,i]` then contains the numbers of nodes belonging to triangle `i`.
- `segmentlist`: $2\times n_{segs}$ integer array describing which two nodes belong to a given boundary segment. `segmentlist[:,i]` contains the numbers of nodes for boundary segment i.


### Triangle form factors


For triangle `itri`, we want to calculate the corresponding form factors $e$ and $\omega$:
"""

# ╔═╡ c1cfe76c-55fa-11eb-2455-01f81a258ab1
function trifactors!(ω, e, itri, pointlist, trianglelist)
    # Obtain the node numbers for triangle itri
	i1=trianglelist[1,itri]
    i2=trianglelist[2,itri]
    i3=trianglelist[3,itri]
	
    # Calculate triangle area: 
    #   Matrix of edge vectors
    V11= pointlist[1,i2]- pointlist[1,i1]
    V21= pointlist[2,i2]- pointlist[2,i1]
    
    V12= pointlist[1,i3]- pointlist[1,i1]
    V22= pointlist[2,i3]- pointlist[2,i1]
    
    V13= pointlist[1,i3]- pointlist[1,i2]
    V23= pointlist[2,i3]- pointlist[2,i2]
    
    #   Compute determinant 
    det=V11*V22 - V12*V21
   
	#   Area
	area=0.5*det
    
    # Squares of edge lengths
    dd1=V13*V13+V23*V23 # l32
    dd2=V12*V12+V22*V22 # l31
    dd3=V11*V11+V21*V21 # l21
        
    # Contributions to e_kl=σ_kl/h_kl
    e[1]= (dd2+dd3-dd1)*0.125/area
    e[2]= (dd3+dd1-dd2)*0.125/area
    e[3]= (dd1+dd2-dd3)*0.125/area
    
    # Contributions to ω_k
    ω[1]= (e[3]*dd3+e[2]*dd2)*0.25
    ω[2]= (e[1]*dd1+e[3]*dd3)*0.25
    ω[3]= (e[2]*dd2+e[1]*dd1)*0.25
end                              

# ╔═╡ f5ab7b3c-55fa-11eb-3c77-1f6f4b1600d3
md"""
### Boundary form factors


Here we need for an interface segment of two points $P_A,P_B$ the contributions to the intersection of the Voronoi cell boundary with the outer boundary which is just the half length: $\gamma_A=\frac12|P_AP_B|, \gamma_B=\frac12|P_AP_B|$
"""

# ╔═╡ 658a0216-55fb-11eb-0680-035a781c93c0
function bfacefactors!(γ,ibface, pointlist, segmentlist)
    i1=segmentlist[1,ibface]
    i2=segmentlist[2,ibface]
    dx=pointlist[1,i1]-pointlist[1,i2]
    dy=pointlist[2,i1]-pointlist[2,i2]
    d=0.5*sqrt(dx*dx+dy*dy)
    γ[1]=d
    γ[2]=d
end

# ╔═╡ 6b810fc4-55fc-11eb-311a-518408042edf
md"""
### Matrix assembly

The matrix assembly consists of two loops, one over all triangles,
and another one over the boundary segments.

The implementation hints at the possibility to work in different space dimensions
"""

# ╔═╡ 1562099a-55fc-11eb-25c0-21e48bc2cb3f
function assemble!(matrix, # System matrix
                   rhs,    # Right hand side vector
		           δ,      # heat conduction coefficient 
                   f::TF, # Source/sink function
		           α,      # boundary transfer coefficient
		           β::TB,  # boundary condition function
                   pointlist,    
                   trianglelist,
                   segmentlist) where {TF, TB}
   
	num_nodes_per_cell=3;
    num_edges_per_cell=3;
    num_nodes_per_bface=2
    ntri=size(trianglelist,2)
	nbface=size(segmentlist,2)
	
    # Local edge-node connectivity
    local_edgenodes=[ 2 3; 3 1; 1 2]'
   
    # Storage for form factors
    e=zeros(num_nodes_per_cell)
    ω=zeros(num_edges_per_cell)
    γ=zeros(num_nodes_per_bface)

    # Initialize right hand side to zero
    rhs.=0.0

    # Loop over all triangles
    for itri=1:ntri
        trifactors!(ω,e,itri,pointlist,trianglelist)

		# Assemble nodal contributions to right hand side
        for k_local=1:num_nodes_per_cell
            k_global=trianglelist[k_local,itri]
            x=pointlist[1,k_global]
            y=pointlist[2,k_global]
            rhs[k_global]+=f(x,y)*ω[k_local]
        end
		
        # Assemble edge contributions to matrix
        for iedge=1:num_edges_per_cell
            k_global=trianglelist[local_edgenodes[1,iedge],itri]
            l_global=trianglelist[local_edgenodes[2,iedge],itri]
            matrix[k_global,k_global]+=δ*e[iedge]
            matrix[l_global,k_global]-=δ*e[iedge]
            matrix[k_global,l_global]-=δ*e[iedge]
            matrix[l_global,l_global]+=δ*e[iedge]
        end
    end

    # Assemble boundary conditions
    
    for ibface=1:nbface
		bfacefactors!(γ,ibface, pointlist, segmentlist)
        for k_local=1:num_nodes_per_bface
            k_global=segmentlist[k_local,ibface]
            matrix[k_global,k_global]+=α*γ[k_local]
            x=pointlist[1,k_global]
            y=pointlist[2,k_global]
            rhs[k_global]+=β(x,y)*γ[k_local]
        end
    end
end


# ╔═╡ 05d15de4-55fe-11eb-0d1f-a3e5cbbc167f
md"""
### Graphical representation

It would be nice to have a graphical representation of the solution data. We can interpret the solution as a piecewise linear function on the triangulation: each triangle has three nodes each carrying one solution value. 

On the other hand, a linear function of two variables is defined by values in three points. This allows to define a piecewise linear, continuous solution function. This approach is well known for the finite element method which we will introduce later.  
"""

# ╔═╡ 5acfe212-560c-11eb-0405-5187d8729a88
md"""
An alternative way of showing the result is the 3D plot of the function graph:
"""

# ╔═╡ 8a557f46-55fe-11eb-2daf-dfd8b6228212
md"""
## Calculation example

Now we are able to solve our intended problem.

### Grid generation
"""

# ╔═╡ adb2d93e-55fe-11eb-021e-377f2cc0804a
function make_grid(;maxarea=0.01)
	triin=TriangulateIO()
    triin.pointlist=Matrix{Cdouble}([-1.0 -1.0; 1.0 -1.0 ; 1.0 1.0 ; -1.0 1.0 ]')
    triin.segmentlist=Matrix{Cint}([1 2 ; 2 3 ; 3 4 ; 4 1 ]')
    triin.segmentmarkerlist=Vector{Int32}([1, 2, 3, 4])
	a=@sprintf("%f",maxarea)
    (triout, vorout)=triangulate("pqAa$(a)qQD", triin)
	triin, triout
end

# ╔═╡ 8916c17a-5609-11eb-2308-d545e4d83556
md"""
### Plotting the grid
In the triout data structure, we indeed see a `pointlist`, a `trianglelist` and a `segmentlist`.

We use the `plot_in_out` function from Triangulate.jl to plot the grid.

Plot grid: $(@bind do_plot_grid CheckBox(true))
"""

# ╔═╡ d2674a6a-5600-11eb-0f25-53e931f4b907
md"""
#### Desired number of triangles
From the desired number of triangles, we can claculate a value fo the maximum area constraint passed to the mesh generator:
Desired number of triangles: $(@bind desired_number_of_triangles Slider(10:10:10000,default=20,show_value=true))
"""

# ╔═╡ e5c4681a-55fe-11eb-2148-1568148c2d51
triin,triout=make_grid(maxarea=4.0/desired_number_of_triangles)

# ╔═╡ 133bafac-55fe-11eb-09b1-ad656d81eda2
function plot(u, pointlist, trianglelist)
    cmap="coolwarm" # color map for color coding function values
    num_isolines=10 # number of isolines for plot
    ax=gca(); ax.set_aspect(1) # don't distort the plot

	# bring data into format understood by PyPlot
    x=view(pointlist,1,:)
    y=view(pointlist,2,:)
    t=transpose(triout.trianglelist.-1)

	# Many (50) filled contour lines give the impression of a smooth color scale
    tricontourf(x,y,t,u,levels=50,cmap=cmap)
    colorbar(shrink=0.5) # Put a color bar next to the plot
	
	# Overlay the plot with isolines
    tricontour(x,y,t,u,levels=num_isolines,colors="k")
end


# ╔═╡ 69bf5f6c-560c-11eb-0ec0-0590da524791
function plot3d(u, pointlist, trianglelist)
    cmap="coolwarm" 
	fig=figure(2)
    x=view(pointlist,1,:)
    y=view(pointlist,2,:)
    t=transpose(triout.trianglelist.-1)
    plot_trisurf(x,y,t,u,cmap=cmap)
end

# ╔═╡ dd7ff76e-55fe-11eb-37f2-e195a5c506d5
if do_plot_grid
	clf()
	plot_in_out(PyPlot,triin,triout)
	gcf().set_size_inches(5,5);
	gcf()
end

# ╔═╡ 82ac4d12-5601-11eb-3360-4b11084f974a
md"""
Number of points: $(size(triout.pointlist,2)), number of triangles:  $(size(triout.trianglelist,2)).
"""

# ╔═╡ d9b59d94-560a-11eb-01cb-3fbf15759112
md"""
### Solving the problem

#### Problem data
"""

# ╔═╡ ab4ba264-560a-11eb-18df-2d42c80ce562
f(x,y)=sinpi(x)*cospi(y)

# ╔═╡ af4ea19a-560a-11eb-2429-4bd8a3d7516e
 β(x,y)=0

# ╔═╡ 32ba2626-560b-11eb-0978-15e42055e956
md"""
δ: $(@bind δ Slider(0.1:0.1:100,default=1.0,show_value=true))
α: $(@bind α Slider(0.0:0.1:100,default=1.0,show_value=true))
"""

# ╔═╡ 88b18360-55fe-11eb-240f-4b91d97414aa
function solve_example(triout)
    n=size(triout.pointlist,2)
    matrix=spzeros(n,n)
    rhs=zeros(n)
    assemble!(matrix,rhs,δ,f,α,β,triout.pointlist,triout.trianglelist, triout.segmentlist)
    sol=(matrix)\rhs
end 

# ╔═╡ b8dec6e8-55ff-11eb-3a17-d1af02a600a6
solution=solve_example(triout)

# ╔═╡ a7de78dc-55ff-11eb-0973-3535cdbdbca3
clf(); plot(solution,triout.pointlist, triout.trianglelist);gcf().set_size_inches(4,4);gcf()

# ╔═╡ 92e11dc8-560f-11eb-34c0-89714a0c2ad6
md"""
3D Plot ?
$@bind do_3d_plot CheckBox(default=false))
"""

# ╔═╡ ccb69a9c-560c-11eb-05ff-bb744edf5564
if do_3d_plot
	clf(); plot3d(solution,triout.pointlist, triout.trianglelist);gcf().set_size_inches(4,4);gcf()
end

# ╔═╡ 1f8c1dee-5551-4aef-9454-9f69fd0c82b1
html""" <hr>"""

# ╔═╡ b3138cac-7c2f-40c7-a68e-405f008fa632
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
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
Triangulate = "f7e6ffb2-c36d-4f8f-a77e-16e897189344"

[compat]
HypertextLiteral = "~0.9.3"
PlutoUI = "~0.7.21"
PyPlot = "~2.10.0"
Triangulate = "~2.1.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "abb72771fd8895a7ebd83d5632dc4b989b022b5b"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.2"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "6cdc8832ba11c7695f494c9d9a1c31e90959ce0f"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.6.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "b68904528fd538f1cb6a3fbc44d2abdc498f9e8e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.21"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "4ba3651d33ef76e24fef6a598b63ffd1c5e1cd17"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.92.5"

[[PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "14c1b795b9d764e1784713941e787e1384268103"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.10.0"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[Triangle_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bfdd9ef1004eb9d407af935a6f36a4e0af711369"
uuid = "5639c1d2-226c-5e70-8d55-b3095415a16a"
version = "1.6.1+0"

[[Triangulate]]
deps = ["DocStringExtensions", "Libdl", "Printf", "Test", "Triangle_jll"]
git-tree-sha1 = "2b4f716b192c0c615d96d541ee029e85666388cb"
uuid = "f7e6ffb2-c36d-4f8f-a77e-16e897189344"
version = "2.1.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VersionParsing]]
git-tree-sha1 = "e575cf85535c7c3292b4d89d89cc29e8c3098e47"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.1"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─bb74bef3-dab3-46e5-93f4-464ccf5d558c
# ╠═60941eaa-1aea-11eb-1277-97b991548781
# ╟─d1edd6da-55f6-11eb-2411-8f7d467a6ce3
# ╟─9eae02a6-5603-11eb-1384-7f81fe7e4a3d
# ╟─e7ff17f0-55f3-11eb-3dbe-31037e82efd9
# ╟─8eef4836-55f2-11eb-0ae1-a1e54cf4163d
# ╟─b8d88140-55ec-11eb-1ce0-b139f9bcc46f
# ╟─1a7f1e8e-55f4-11eb-3a18-7f34aca5b559
# ╟─6209efdc-55f8-11eb-1fa5-49545391945d
# ╟─afc9f670-5604-11eb-23c9-cd64f9f404b8
# ╟─1bcbe0aa-55fa-11eb-13ea-6906b6fcae24
# ╠═c1cfe76c-55fa-11eb-2455-01f81a258ab1
# ╟─f5ab7b3c-55fa-11eb-3c77-1f6f4b1600d3
# ╠═658a0216-55fb-11eb-0680-035a781c93c0
# ╟─6b810fc4-55fc-11eb-311a-518408042edf
# ╠═1562099a-55fc-11eb-25c0-21e48bc2cb3f
# ╟─05d15de4-55fe-11eb-0d1f-a3e5cbbc167f
# ╠═133bafac-55fe-11eb-09b1-ad656d81eda2
# ╟─5acfe212-560c-11eb-0405-5187d8729a88
# ╠═69bf5f6c-560c-11eb-0ec0-0590da524791
# ╟─8a557f46-55fe-11eb-2daf-dfd8b6228212
# ╠═adb2d93e-55fe-11eb-021e-377f2cc0804a
# ╠═e5c4681a-55fe-11eb-2148-1568148c2d51
# ╟─8916c17a-5609-11eb-2308-d545e4d83556
# ╟─dd7ff76e-55fe-11eb-37f2-e195a5c506d5
# ╟─82ac4d12-5601-11eb-3360-4b11084f974a
# ╠═d2674a6a-5600-11eb-0f25-53e931f4b907
# ╟─d9b59d94-560a-11eb-01cb-3fbf15759112
# ╠═ab4ba264-560a-11eb-18df-2d42c80ce562
# ╠═af4ea19a-560a-11eb-2429-4bd8a3d7516e
# ╟─32ba2626-560b-11eb-0978-15e42055e956
# ╠═88b18360-55fe-11eb-240f-4b91d97414aa
# ╠═b8dec6e8-55ff-11eb-3a17-d1af02a600a6
# ╠═a7de78dc-55ff-11eb-0973-3535cdbdbca3
# ╟─92e11dc8-560f-11eb-34c0-89714a0c2ad6
# ╠═ccb69a9c-560c-11eb-05ff-bb744edf5564
# ╟─1f8c1dee-5551-4aef-9454-9f69fd0c82b1
# ╟─b3138cac-7c2f-40c7-a68e-405f008fa632
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
