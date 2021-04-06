from fenics import *
from mshr import *
import numpy as np
import time 
import os
os.environ['OMP_NUM_THREADS'] = '1'

tf = 150           # final time
num_steps = 1800000 # number of time steps
dt = tf / num_steps # time step size
mu = 0.008*100   # dynamic viscosity
rho = 1300            # density
force = 100

mu0  = Constant(1.25*10**(-6)) # N/A^2
bettaR = Constant(5.6*10**(-4)) # 1/K
chi0 = Constant(0.1257)#Constant(12.57)
T0 = Constant(293) #K




mesh = Mesh('omega_mesh.xml')



# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)
QF = VectorFunctionSpace(mesh, 'P', 2)
QT = FunctionSpace(mesh, 'P', 1)
QH = VectorFunctionSpace(mesh, 'P', 2)

#OLD
####################################################
#Define boundary
#heat sink and source temperature
high_temp = 'on_boundary && x[0]>-0.015 && x[0]<-0.005 && x[1]>0.00863 && x[1]<0.01712'
low_temp  = 'on_boundary && x[0]>-0.0025 && x[0]<0.0025 && x[1]>-0.01712 && x[1]<-0.00863' 
# ambient temperature outer boundary
reg_temp1 = 'on_boundary && x[0]>=-0.005 && x[0]<0.015 && x[1]>0.00863 && x[1]<0.01712'
reg_temp2 = 'on_boundary && x[0]>=0.015 && x[0]<0.0203 && x[1]>-0.01712 && x[1]<0.01712'
reg_temp3 = 'on_boundary && x[0]>-0.0203 && x[0]<=-0.015 && x[1]>-0.01712 && x[1]<0.01712'
reg_temp4 = 'on_boundary && x[0]>-0.015 && x[0]<=-0.0025 && x[1]>-0.01712 && x[1]<-0.00863' 
reg_temp5 = 'on_boundary && x[0]>=0.0025 && x[0]<0.015 && x[1]>-0.01712 && x[1]<-0.00863' 
# ambient temperature inner boundary
reg_Inside = 'on_boundary && x[0]>-0.0118 && x[0]<0.0118 && x[1]>-0.00862 && x[1]<0.00862'

# full wall 
full_wall = 'on_boundary && x[0]>-0.0203 && x[0]<0.0203 && x[1]>-0.01712 && x[1]<0.01712'

# Define boundary conditions
bcT_inside = DirichletBC(QT, Constant(293), reg_Inside)
bcT_hot = DirichletBC(QT, Constant(328), high_temp)
bcT_cold = DirichletBC(QT, Constant(279), low_temp)
bcT_out1 = DirichletBC(QT, Constant(293), reg_temp1)
bcT_out2 = DirichletBC(QT, Constant(293), reg_temp2)
bcT_out3 = DirichletBC(QT, Constant(293), reg_temp3)
bcT_out4 = DirichletBC(QT, Constant(293), reg_temp4)
bcT_out5 = DirichletBC(QT, Constant(293), reg_temp5)
bcT = [bcT_inside, bcT_hot, bcT_cold, bcT_out1, bcT_out2, bcT_out3, bcT_out4, bcT_out5 ]
#######################################################


bcu_inside = DirichletBC(V, Constant((0,0)), reg_Inside)
bcu_hot = DirichletBC(V, Constant((0,0)), high_temp)
bcu_cold = DirichletBC(V, Constant((0,0)), low_temp)
bcu_out1 = DirichletBC(V, Constant((0,0)), reg_temp1)
bcu_out2 = DirichletBC(V, Constant((0,0)), reg_temp2)
bcu_out3 = DirichletBC(V, Constant((0,0)), reg_temp3)
bcu_out4 = DirichletBC(V, Constant((0,0)), reg_temp4)
bcu_out5 = DirichletBC(V, Constant((0,0)), reg_temp5)
bcu = [bcu_inside, bcu_hot, bcu_cold,
       bcu_out1, bcu_out2, bcu_out3, bcu_out4, bcu_out5 ]

bcp_outer = DirichletBC(Q, Constant(0), full_wall)
bcp_inner = DirichletBC(Q, Constant(0), reg_Inside)
bcp = []

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)
vT = TestFunction(QT)


# Define functions for solutions at previous and current time steps
u_n = Function(V)
u_  = Function(V)
p_n = Function(Q)
p_  = Function(Q)
T = Function(QT)
T_n = Function(QT)

#initial temperature
Ti = TrialFunction(QT)
Tiv = TestFunction(QT)
Tif = Constant(0)
aTi = dot(grad(Ti), grad(Tiv))*dx
LTi = Tif*Tiv*dx
# Compute solution
Ti = Function(QT)
solve(aTi == LTi, Ti, bcT)
T_n = Ti

# magnetic field
m1 = Constant(0.65)
sx1 = Constant(-0.03)
sy1 = Constant(0.02)

H_expr = Expression(('m*(2 * pow(x[0] - sx, 2) - pow(x[1] - sy, 2))/(4*pi*pow(pow(x[0] - sx,2)+pow(x[1] - sy,2),2.5))',
                     'm*(3*(x[0] - sx)*(x[1] - sy))/(4*pi*pow(pow(x[0] - sx,2)+pow(x[1] - sy,2),2.5))')
                    ,degree = 2, m = m1, sx = sx1, sy = sy1)
H = interpolate(H_expr, V)
H = interpolate(H_expr, QH)

#define Kelvin body force
def chi_m(T):
    return chi0/(1 + bettaR*(T-T0))

def Kelvin_f(T, H):
    return mu0 * chi_m(T)*(0.5*(1+chi_m(T))*grad(dot(H,H))
                           -bettaR*(chi_m(T)**2/chi0)*H*dot(H, grad(T))) 
fi = Function(QF)
fi = Kelvin_f(T_n, H)
f = project(fi, QF)

# Define expressions used in variational forms
U  = 0.5*(u_n + u)
n  = FacetNormal(mesh)
#f  = Constant((-force, -force))
k  = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)

# Define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# Define variational problem for step 1
F1 = (rho*dot((u - u_n) / k, v)*dx  
   + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx  
   + inner(sigma(U, p_n), epsilon(v))*dx  
   + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds  
   - dot(f, v)*dx)
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

# Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

# Define variational problem for convection
lam = Constant(0.06)
cp = Constant(4180)

# Define variational problem
# Define variational problem for heat convection
F4 = (((T - T_n) / k)*vT*dx  + dot(u_, grad(T))*vT*dx
         + (lam / (cp*rho))*dot(grad(T), grad(vT))*dx)


# Create XDMF files for visualization output
xdmffile_u = XDMFFile('navier_stokes_cavity/velocity.xdmf')
xdmffile_p = XDMFFile('navier_stokes_cavity/pressure.xdmf')
xdmffile_T = XDMFFile('navier_stokes_cavity/temperature.xdmf')
xdmffile_F = XDMFFile('navier_stokes_cavity/Kelvin_body_force.xdmf')



vtkfile_u = File('navier_stokes_cavity/velocity.pvd')
vtkfile_p = File('navier_stokes_cavity/pressure.pvd')
vtkfile_T = File('navier_stokes_cavity/temperature.pvd')
vtkfile_F = File('navier_stokes_cavity/Kelvin_body_force.pvd')




# Time-stepping
t = 0
startTime = time.time()
for n in range(num_steps):

    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    A1 = assemble(a1)
    [bc.apply(A1) for bc in bcu]
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1,  'bicgstab', 'hypre_amg')

    # Step 2: Pressure correction step
    solve(a2 == L2, p_, bcp)

    # Step 3: Velocity correction step
    A3 = assemble(a3)
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3, 'cg', 'sor')

    # solve for temperature 
    solve(F4 == 0, T, bcT)
    
    # Save solution to file (XDMF/HDF5)
    if (n % 100 == 0): 
        xdmffile_u.write(u_, t)
        xdmffile_p.write(p_, t)
        xdmffile_T.write(T, t)
        xdmffile_F.write(f, t)

        # Save solution to file (PVD)
        vtkfile_u << (u_, t)
        vtkfile_p << (p_, t)
        vtkfile_T << (T, t)
        vtkfile_F << (f, t)



    #Update Kelvin body force
    fi = Kelvin_f(T, H)
    f.assign(project(fi, QF))
    
    
    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)
    T_n.assign(T)


    print(n)
    
doneTime = time.time()
elapsed = doneTime - startTime
print(elapsed)
