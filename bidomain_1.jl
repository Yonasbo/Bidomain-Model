begin 
    using ExtendableGrids, VoronoiFVM,LinearAlgebra,JSON3,JLD2,DifferentialEquations,Sundials
	
end


begin
	ε=.2 # scale reaction functions
	γ=0.5 # constant for reaction function g
	β=1.0 # constant for reaction function g
end

function storage(f,u,node,data)
	f[1]=u[1]
	f[2]=0
	f[3]=u[3]
end

function diffusion(f,u,edge,data)
	σ_i=1.0
	σ_e=1.0
	f[1]=σ_i*(u[1,1]-u[1,2]+u[2,1]-u[2,2])
	f[2]=(σ_i*(u[1,1]-u[1,2])+(σ_i+σ_e)*(u[2,1]-u[2,2]))
end

function angle(a, b)
    return acosd(clamp(a⋅b/(norm(a)*norm(b)), -1, 1))
end

function pellipse(σ_1,σ_2,ϕ)
	return sqrt(σ_1^2*cosd(ϕ)^2+σ_2^2*sind(ϕ)^2)
end

function diffusion_2d(f,u,edge,data)
	
	σ_i=25*Diagonal([ 0.263, 0.0263])
	σ_e=25*Diagonal([ 0.263, 0.1087])
	vec_1=vec([edge[1,1] edge[2,1]])
	vec_2=vec([edge[1,2] edge[2,2]])
	
	#D_i=norm(σ_i*(vec_1-vec_2))/norm(vec_1-vec_2)
	#D_e=norm(σ_e*(vec_1-vec_2))/norm(vec_1-vec_2)
	ϕ=angle((vec_1-vec_2),vec([1 0]))
	D_i=pellipse(25*0.263,25*0.0263,ϕ)
	D_e=pellipse(25*0.263,25*0.1087,ϕ)
	
	f[1]=D_i*(u[1,1]-u[1,2]+u[2,1]-u[2,2])
	f[2]=(D_i*(u[1,1]-u[1,2])+(D_i+D_e)*(u[2,1]-u[2,2]))
end

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

function reaction(f,u,node,data)
    f[1]= -1/ε*(u[1]-u[1]^3/3-u[3])
    f[3]= -ε*(u[1]+β-γ*u[3])
end

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
end

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
    h_3d=1.4) 
### check for dimension to choose 1D or 2D grid,diffusion and boundary conditions.
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
control.max_lureuse=10
#control.verbose=true
control.Δt_max=0.01*tend
control.Δu_opt=2
tsol=VoronoiFVM.solve(system,inival=inival,times=[0,tend],control=control,log=true,damp=damp,damp_grow=damp_grow)
#problem = ODEProblem(system,inival,(0,tend))
#odesol = DifferentialEquations.solve(problem,ImplicitEuler())
#tsol=reshape(odesol,system)
return tsol,grid,tend,dim,system
end

begin
    tsol_2d,grid_2d,tend_2d,dim_2d,system_2d=bidomain(dim=2,
		tend=50,
		gridlength=70,
		tstep=1.0e-3,
		damp=0.5,
		damp_grow=1.8,
		h_2d=0.466)
    
    jldsave("testoo.jld2"; tsol_2d,grid_2d,tend_2d,dim_2d,system_2d)
    end