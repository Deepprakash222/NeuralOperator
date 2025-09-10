import json
import ufl
import dolfin as dl
from datetime import datetime
#sys.path.append(os.environ.get('HIPPYLIB_PATH', "../../"))
import hippylib as hp
#sys.path.append(os.environ.get('HIPPYFLOW_PATH',"../../"))
import hippyflow as hf

from helpers import *
from generate_samples import *
from train_nn import *
import arviz as az
from pyro.infer import MCMC, NUTS, Predictive, EmpiricalMarginal
from pyro.infer.autoguide import init_to_mean, init_to_median, init_to_value
from pyro.infer.mcmc.util import summary

import pandas as pd
def pyro_model(interpolation_input_ ,num_layers, NN_model,Interpolation_matrix, u_shift, phi, obs_data, device ):
        
            """
            This Pyro model represents the probabilistic aspects of the geological model.
            It defines a prior distribution for the top layer's location and
            computes the thickness of the geological layer as an observed variable.

            
            interpolation_input_: represents the dictionary of random variables for surface parameters
            
            num_layers: represents the number of layers we want to include in the model
            
            """

            
            parameter = []
            
            # Create a random variable based on the provided dictionary used to modify input data of gempy
            counter=1
            for interpolation_input_data in interpolation_input_[:num_layers]:
                
                # Check if user wants to create random variable based on modifying the surface points of gempy
                if interpolation_input_data["update"]=="interface_data":
                    # Check what kind of distribution is needed
                    if interpolation_input_data["prior_distribution"]=="normal":
                        mean = interpolation_input_data["normal"]["mean"]
                        std  = interpolation_input_data["normal"]["std"]
                        parameter.append(pyro.sample("mu_"+ str(counter), dist.Normal(mean, std)).to(device))
                        
                    elif interpolation_input_data["prior_distribution"]=="uniform":
                        min = interpolation_input_data["uniform"]["min"]
                        max = interpolation_input_data["uniform"]["min"]
                        parameter.append(pyro.sample("mu_"+ str(interpolation_input_data['id']), dist.Uniform(min, max)).to(device))

                        
                    else:
                        print("We have to include the distribution")
                
                counter=counter+1
             
            input_data = torch.tensor(parameter, device=device)
            NN_model.eval()
            NN_output = NN_model(input_data)
            output =torch.matmul(Interpolation_matrix, torch.matmul(NN_output, phi.T) + u_shift)
            #print(output.shape)
            #output = NN_output
            
            with pyro.plate("likelihood", obs_data.shape[0]):
                pyro.sample("obs", dist.Normal(output, 0.05), obs=obs_data).to(device)
            
            

def main():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        #device = torch.device("mps")
        device = torch.device("cpu")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    nx =31; ny = 31
    nodes = (nx+1)*(ny+1)
    mesh = dl.RectangleMesh(dl.Point(0.0, 0.0), dl.Point(1.0, 1.0), nx, ny)
    Vh_STATE = dl.FunctionSpace(mesh, "CG", 2)
    Vh_PARAMETER = dl.FunctionSpace(mesh, "CG", 1)
    Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE]

    d2v = dl.dof_to_vertex_map(Vh[hp.PARAMETER])
    v2d = dl.vertex_to_dof_map(Vh[hp.PARAMETER])


    def u_boundary(x, on_boundary):
        return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

    u_bdr = dl.Expression("x[1]", degree=1)
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh[hp.STATE], u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh[hp.STATE], u_bdr0, u_boundary)

    f = dl.Constant(0.0)
    #f = dl.Expression("sin(2*pi*x[0]) * exp(x[1]+ x[0])", degree=5)

    def pde_varf(u,m,p):
        return m*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx - f*p*ufl.dx
        

    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

    Borehole_location = [0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    Borehole_extent = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    Borehole_points = [20, 20, 20, 20, 20, 20, 20, 20, 20]
    target_list=[]
    for i, x in enumerate(Borehole_location):
        z_data = np.linspace(Borehole_extent[i], 0.9, Borehole_points[i])
        first_column = np.full((z_data.shape[0],), x)
        two_d_array = np.column_stack((first_column, z_data))
        target_list.append(two_d_array)
    targets = np.vstack(target_list)
    
    B = hp.assemblePointwiseObservation(Vh[hp.STATE], targets)
    
    u_trial = dl.TrialFunction(Vh[hp.STATE])
    u_test = dl.TestFunction(Vh[hp.STATE])

    M_U = dl.assemble(dl.inner(u_trial,u_test)*dl.dx)

    I_U = hf.StateSpaceIdentityOperator(M_U)
    #observable = hf.LinearStateObservable(pde,M_U)
    observable_1 = hf.LinearStateObservable(pde,I_U)
    observable_2 = hf.LinearStateObservable(pde,B)

    Jm_1 = hf.ObservableJacobian(observable_1)
    Jm_2 = hf.ObservableJacobian(observable_2)

    m_trial = dl.TrialFunction(Vh[hp.PARAMETER])
    m_test = dl.TestFunction(Vh[hp.PARAMETER])

    M_M = dl.assemble(dl.inner(m_trial,m_test)*dl.dx)
    
    m_initial, sp_coords_copy_test, gempy_model= create_true_data(mesh=mesh, nodes=nodes, filename=None)
    m_initial = 2 * m_initial
    
    m = dl.Function(Vh[hp.PARAMETER])
    m.vector().set_local(m_initial[d2v])
    
    # plt.axis("off")
    # col = dl.plot(m,cmap="viridis" )
    # fig = plt.gcf()
    # fig.colorbar(col) 
    # # Add axis labels and title
    # plt.xlabel("x-coordinate")
    # plt.ylabel("z-coordinate")
    # plt.title("Parameter Field $m(x,z)$")

    # plt.tight_layout()
    # fig.set_size_inches(6, 6)
    # plt.show()
    fig, ax = plt.subplots(figsize=(6,6))  # Create new figure explicitly
    im =plt.imshow(m_initial.reshape((nx+1,nx+1)),
                   cmap='viridis',
                   origin='lower',
                   interpolation='bilinear',
                   extent=[0, 1, 0, 1])
    # Add colorbar to this figure
    fig.colorbar(im, ax=ax)
    # Add axis labels and title
    ax.set_xlabel("x-coordinate")
    ax.set_ylabel("z-coordinate")
    ax.set_title("Parameter Field $m(x,z)$")
    plt.tight_layout()
    plt.show()
    
    u = dl.Function(Vh[hp.STATE])
    uadj = dl.Function(Vh[hp.ADJOINT])
    
    dU = u.vector().get_local().shape[0]
    dO = targets.shape[0]
    u_data = np.zeros(dU)
    u_obs  = np.zeros(dO)
    
    x = [u.vector(),m.vector(),uadj.vector()]
    pde.solveFwd(x[hp.STATE], x)
    
    
    # Get the data
    u_true = x[hp.STATE].get_local()
    u_data = B.array() @ u_true
    u_obs = observable_2.evalu(x[hp.STATE]).get_local()
    A = u_obs - u_data
    A[np.abs(A)<1e-10]=0
    print(np.sum(A))
    
    # plot u_true
    u.vector().set_local(u_true)
    # plt.axis("off")
    # col = dl.plot(u,cmap="viridis" )
    # fig = plt.gcf()
    # fig.colorbar(col) 
    # # Add axis labels and title
    # plt.scatter(targets[:,0], targets[:,1], color='white', marker='s', s=10, label='Observations')
    # plt.legend()
    # plt.xlabel("x-coordinate")
    # plt.ylabel("z-coordinate")
    # plt.title("State Field $u(x,z)$")

    # plt.tight_layout()
    # fig.set_size_inches(6, 6)
    # plt.show()
    
    
    # noise level
    noise_level = 0.05


    ud = dl.Function(Vh[hp.STATE])
    ud.assign(u)

    # perturb state solution and create synthetic measurements ud
    # ud = u + ||u||/SNR * random.normal
    MAX = ud.vector().norm("linf")

    # Create noise vector and insert the numpy array
    noise = dl.Vector()
    M_U.init_vector(noise, 1)
    hp.parRandom.normal(noise_level * MAX, noise)

    bc0.apply(noise)

    ud.vector().axpy(1., noise)

    # plot
    hp.nb.multi1_plot([u, ud], ["State Field", "Synthetic observations"])
    # Add axis labels and title
    plt.scatter(targets[:,0], targets[:,1], color='white', marker='s', s=10, label='Observations')
    plt.legend()
    plt.xlabel("x-coordinate")
    plt.ylabel("z-coordinate")
    plt.title("Synthetic observations with measurement points")
    plt.show()
    
    # Generate Gaussian noise
    mean = 0
    std_dev = 0.05
    noise = np.random.normal(loc=mean, scale=std_dev, size=u_obs.shape)
    u_obs = u_obs + noise
    

    
    dtype =torch.float32
    Mphi_r= torch.tensor(np.load('./saved_model/Mphi_r.npy'), dtype=dtype, device=device)
    phi_r = torch.tensor(np.load('./saved_model/phi_r.npy'), dtype=dtype, device=device)
    u_shift = torch.tensor(np.load('./saved_model/u_shift.npy'), dtype=dtype, device=device)
    
    # Load the full model directly
    model_without_jacobian = torch.load("./saved_model/model_without_jacobian.pth",map_location=device,weights_only=False)
    model_jacobian_full    = torch.load("./saved_model/model_jacobian_full.pth",map_location=device, weights_only=False)
    model_jacobian_truncated    = torch.load("./saved_model/model_jacobian_truncated.pth",map_location=device, weights_only=False)

    ###############################################################################
    # Make a list of gempy parameter which would be treated as a random variable
    ###############################################################################

    test_list=[]
    std = 0.03  # 0.125 , 4*std = gap between two layers
    test_list.append({"update":"interface_data","id":torch.tensor([1]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[1,2],dtype=dtype, device=device), "std":torch.tensor(std,dtype=dtype, device=device)}})
    
    num_layers = len(test_list) 
    
    model = model_jacobian_full
    Interpolation_matrix = torch.tensor(B.array(), dtype=dtype, device=device)
    #obs_data = torch.tensor(ud.vector().get_local(), dtype=dtype, device=device)
    obs_data = torch.tensor(u_obs, dtype=dtype, device=device)
    #print(obs_data.shape)
    dot = pyro.render_model(pyro_model, model_args=(test_list, num_layers, model, Interpolation_matrix,  u_shift, phi_r, obs_data , device),render_distributions=True)
    pyro.set_rng_seed(42)

    prior = Predictive(pyro_model,num_samples=10000)(test_list, num_layers, model, Interpolation_matrix,   u_shift, phi_r, obs_data , device)
    #plt.figure(figsize=(8,10))
    data = az.from_pyro(prior=prior)
    #az.plot_trace(data.prior)
    
    ################################################################################
    # Posterior
    ################################################################################
    pyro.primitives.enable_validation(is_validate=True)
    nuts_kernel = NUTS(pyro_model, step_size=0.0085, adapt_step_size=True, target_accept_prob=0.9, max_tree_depth=10, init_strategy=init_to_mean)
    mcmc = MCMC(nuts_kernel, num_samples=100, mp_context="spawn", warmup_steps=100,num_chains=5, disable_validation=True)
    mcmc.run(test_list, num_layers, model,Interpolation_matrix,  u_shift, phi_r, obs_data , device)
    posterior_samples = mcmc.get_samples(group_by_chain=False)
    samples = mcmc.get_samples(group_by_chain=True)
    
    summary_stats = summary(samples)
    #print(summary_stats) 
    # Convert to a Pandas DataFrame
    df = pd.DataFrame.from_dict(summary_stats, orient="index")
    # This transposes the dictionary so each parameter is a row

    # Now save to CSV
    df.to_csv("pyro_summary.csv")
    
    list_parameter_mean = []
    list_parameter_mean_plus_std = []
    list_parameter_mean_minus_std = []
    for index, (key, values) in enumerate(posterior_samples.items()):
        print("Prior mean: ", test_list[index]["normal"]["mean"], "Prior std: ", test_list[index]["normal"]["std"])
        print("Posterior mean: ",torch.mean(values), "Posterior std: ",torch.std(values))
        list_parameter_mean.append(torch.mean(values).to(torch.float64))
        list_parameter_mean_plus_std.append(torch.mean(values).to(torch.float64) + torch.std(values).to(torch.float64))
        list_parameter_mean_minus_std.append(torch.mean(values).to(torch.float64) - torch.std(values).to(torch.float64))
    
    posterior_predictive = Predictive(pyro_model, posterior_samples)(test_list, num_layers, model, Interpolation_matrix,  u_shift, phi_r, obs_data , device)
    plt.figure(figsize=(8,10))
    data = az.from_pyro(posterior=mcmc, prior=prior, posterior_predictive=posterior_predictive)
    az.plot_trace(data)
    ###############################################TODO################################
    # Plot and save the file for each parameter
    ###################################################################################
    for i in range(len(test_list)):
        plt.figure(figsize=(8,10))
        az.plot_density(
        data=[data.posterior, data.prior],
        shade=.9,
        bw=0.003,  # increase bandwidth for smoother curve
        var_names=['mu_' +str(i+1)],
        data_labels=["Posterior Predictive", "Prior Predictive"],
        )
        plt.savefig("./saved_model/mu_"+ str(i)+".png")
        plt.close()

    generate_final_model(geo_model=gempy_model, interpolation_input_=test_list, num_layers=num_layers, posterior_data=list_parameter_mean, slope=200, filename='posterior_model.png')
    generate_final_model(geo_model=gempy_model, interpolation_input_=test_list, num_layers=num_layers, posterior_data=list_parameter_mean_plus_std, slope=200, filename='posterior_model_plus_std.png')
    generate_final_model(geo_model=gempy_model, interpolation_input_=test_list, num_layers=num_layers, posterior_data=list_parameter_mean_minus_std, slope=200, filename='posterior_model_minus_std.png')
    
if __name__ == "__main__":
    
    # Your main script code starts here
    print("Script started...")
    
    # Record the start time
    start_time = datetime.now()

    main()
    # Record the end time
    end_time = datetime.now()

    # Your main script code ends here
    print("Script ended...")
    
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    print(f"Elapsed time: {elapsed_time}")
