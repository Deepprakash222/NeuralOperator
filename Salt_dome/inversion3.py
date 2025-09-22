import os
import json
import math
import numpy as np
import torch
import ufl
import dolfin as dl
import arviz as az
import matplotlib.pyplot as plt
from datetime import datetime

import hippylib as hp
import hippyflow as hf

from helpers import *
from generate_samples import *
from train_nn import *

import pandas as pd
import pyro
from pyro import render_model, clear_param_store, primitives
import pyro.distributions as dist
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.optim import ClippedAdam, Adam

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def pyro_model(interpolation_input_, num_layers, NN_model, Interpolation_matrix, u_shift, phi, obs_data, device):
    """
    Probabilistic model:
      - Prior over geological parameters (mu_i)
      - Push params through NN -> field
      - Interpolate to observation points
      - Gaussian likelihood
    """
    parameter = []
    counter = 1
    for interpolation_input_data in interpolation_input_[:num_layers]:
        if interpolation_input_data["update"] == "interface_data":
            if interpolation_input_data["prior_distribution"] == "normal":
                mean = interpolation_input_data["normal"]["mean"]
                std = interpolation_input_data["normal"]["std"]
                parameter.append(pyro.sample(f"mu_{counter}", dist.Normal(mean, std)).to(device))
            elif interpolation_input_data["prior_distribution"] == "uniform":
                mn = interpolation_input_data["uniform"]["min"]
                mx = interpolation_input_data["uniform"]["max"]
                parameter.append(pyro.sample(f"mu_{interpolation_input_data['id']}", dist.Uniform(mn, mx)).to(device))
            else:
                raise ValueError("Unsupported prior_distribution in interpolation_input_")
        counter += 1

    # keep gradients: stack instead of re-wrapping tensors
    input_data = torch.stack(parameter).to(device)

    NN_model.eval()
    NN_output = NN_model(input_data)
    output = torch.matmul(Interpolation_matrix, torch.matmul(NN_output, phi.T) + u_shift)

    with pyro.plate("likelihood", obs_data.shape[0]):
        pyro.sample("obs", dist.Normal(output, 0.05), obs=obs_data).to(device)

def custom_guide(interpolation_input_, num_layers, NN_model, Interpolation_matrix, u_shift, phi, obs_data, device):
    """
    Probabilistic model:
      - Prior over geological parameters (mu_i)
      - Push params through NN -> field
      - Interpolate to observation points
      - Gaussian likelihood
    """
    parameter = []
    counter = 1
    for interpolation_input_data in interpolation_input_[:num_layers]:
        if interpolation_input_data["update"] == "interface_data":
            if interpolation_input_data["prior_distribution"] == "normal":
                mean = interpolation_input_data["normal"]["mean"]
                std = interpolation_input_data["normal"]["std"]
                mean_loc = pyro.param(f"mu_{counter}_loc", mean - 1.5 * std, constraint=dist.constraints.real)
                std_scale = pyro.param(f"mu_{counter}_scale", std * 0.3, constraint=dist.constraints.positive)
                parameter.append(pyro.sample(f"mu_{counter}", dist.Normal(mean_loc, std_scale)).to(device))
            elif interpolation_input_data["prior_distribution"] == "uniform":
                mn = interpolation_input_data["uniform"]["min"]
                mx = interpolation_input_data["uniform"]["max"]
                parameter.append(pyro.sample(f"mu_{interpolation_input_data['id']}", dist.Uniform(mn, mx)).to(device))
            else:
                raise ValueError("Unsupported prior_distribution in interpolation_input_")
        counter += 1


def main():
    # ---- device selection
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("cpu")  # keep CPU for consistency
        print("Using MPS-capable machine; running on CPU for consistency")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    # ---- FEM / PDE setup
    nx = 31
    ny = 31
    nodes = (nx + 1) * (ny + 1)
    mesh = dl.RectangleMesh(dl.Point(0.0, 0.0), dl.Point(1.0, 1.0), nx, ny)
    Vh_STATE = dl.FunctionSpace(mesh, "CG", 2)
    Vh_PARAMETER = dl.FunctionSpace(mesh, "CG", 1)
    Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE]

    d2v = dl.dof_to_vertex_map(Vh[hp.PARAMETER])
    v2d = dl.vertex_to_dof_map(Vh[hp.PARAMETER])

    def u_boundary(x, on_boundary):
        return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

    u_bdr = dl.Expression("x[1]", degree=1)
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh[hp.STATE], u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh[hp.STATE], u_bdr0, u_boundary)

    f = dl.Constant(0.0)

    def pde_varf(u, m, p):
        return m * ufl.inner(ufl.grad(u), ufl.grad(p)) * ufl.dx - f * p * ufl.dx

    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

    # ---- observation locations (boreholes)
    Borehole_location = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    Borehole_extent = [0.1] * 9
    Borehole_points = [50] * 9
    target_list = []
    for i, x in enumerate(Borehole_location):
        z_data = np.linspace(Borehole_extent[i], 0.9, Borehole_points[i])
        first_column = np.full((z_data.shape[0],), x)
        two_d_array = np.column_stack((first_column, z_data))
        target_list.append(two_d_array)
    targets = np.vstack(target_list)

    B = hp.assemblePointwiseObservation(Vh[hp.STATE], targets)

    u_trial = dl.TrialFunction(Vh[hp.STATE])
    u_test = dl.TestFunction(Vh[hp.STATE])

    M_U = dl.assemble(dl.inner(u_trial, u_test) * dl.dx)
    I_U = hf.StateSpaceIdentityOperator(M_U)

    observable_1 = hf.LinearStateObservable(pde, I_U)
    observable_2 = hf.LinearStateObservable(pde, B)

    Jm_1 = hf.ObservableJacobian(observable_1)
    Jm_2 = hf.ObservableJacobian(observable_2)

    m_trial = dl.TrialFunction(Vh[hp.PARAMETER])
    m_test = dl.TestFunction(Vh[hp.PARAMETER])
    M_M = dl.assemble(dl.inner(m_trial, m_test) * dl.dx)

    # ---- synthetic parameter + state
    m_initial, sp_coords_copy_test, gempy_model = create_true_data(mesh=mesh, nodes=nodes, filename=None)
    m_initial = 2 * m_initial

    m = dl.Function(Vh[hp.PARAMETER])
    m.vector().set_local(m_initial[d2v])

    fig, ax = plt.subplots(figsize=(6, 6))
    im = plt.imshow(
        m_initial.reshape((nx + 1, nx + 1)),
        cmap="viridis",
        origin="lower",
        interpolation="bilinear",
        extent=[0, 1, 0, 1],
    )
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("x-coordinate")
    ax.set_ylabel("z-coordinate")
    ax.set_title("Parameter Field $m(x,z)$")
    plt.tight_layout()
    plt.savefig("parameter_field.png")
    plt.close()
    
    u = dl.Function(Vh[hp.STATE])
    uadj = dl.Function(Vh[hp.ADJOINT])

    dU = u.vector().get_local().shape[0]
    dO = targets.shape[0]
    u_data = np.zeros(dU)
    u_obs = np.zeros(dO)

    x = [u.vector(), m.vector(), uadj.vector()]
    pde.solveFwd(x[hp.STATE], x)

    # truth + observations at points
    u_true = x[hp.STATE].get_local()
    u_data = B.array() @ u_true
    u_obs = observable_2.evalu(x[hp.STATE]).get_local()
    A = u_obs - u_data
    A[np.abs(A) < 1e-10] = 0
    print(np.sum(A))

    # noise on state -> synthetic measurements
    noise_level = 0.00
    ud = dl.Function(Vh[hp.STATE])
    ud.assign(u)
    MAX = ud.vector().norm("linf")
    noise = dl.Vector()
    M_U.init_vector(noise, 1)
    hp.parRandom.normal(noise_level * MAX, noise)
    bc0.apply(noise)
    ud.vector().axpy(1.0, noise)

    hp.nb.multi1_plot([u, ud], ["State Field", "Synthetic observations"])
    plt.scatter(targets[:, 0], targets[:, 1], color="white", marker="s", s=10, label="Observations")
    plt.legend()
    plt.xlabel("x-coordinate")
    plt.ylabel("z-coordinate")
    plt.title("Synthetic observations with measurement points")
    plt.savefig("state_field_with_observations.png")
    plt.close()
    
    # add iid Gaussian noise to point observations
    u_obs = u_obs + np.random.normal(loc=0.0, scale=0.05, size=u_obs.shape)

    # ---- load reduced-order objects + NN
    dtype = torch.float32
    Mphi_r = torch.tensor(np.load("./saved_model/Mphi_r.npy"), dtype=dtype, device=device)
    phi_r = torch.tensor(np.load("./saved_model/phi_r.npy"), dtype=dtype, device=device)
    u_shift = torch.tensor(np.load("./saved_model/u_shift.npy"), dtype=dtype, device=device)

    model_without_jacobian = torch.load("./saved_model/model_without_jacobian.pth", map_location=device, weights_only=False)
    model_jacobian_full = torch.load("./saved_model/model_jacobian_full.pth", map_location=device, weights_only=False)
    model_jacobian_truncated = torch.load("./saved_model/model_jacobian_truncated.pth", map_location=device, weights_only=False)

    # ---- priors for Pyro
    test_list = []
    std = 0.03
    test_list.append({
        "update": "interface_data",
        "id": torch.tensor([1]),
        "direction": "Z",
        "prior_distribution": "normal",
        "normal": {
            "mean": torch.tensor(sp_coords_copy_test[1, 2], dtype=dtype, device=device),
            "std": torch.tensor(std, dtype=dtype, device=device),
        },
    })

    num_layers = len(test_list)
    model = model_jacobian_full
    Interpolation_matrix = torch.tensor(B.array(), dtype=dtype, device=device)
    obs_data = torch.tensor(u_obs, dtype=dtype, device=device)

    # optional: render model (guarded)
    try:
        render_model(
            pyro_model,
            model_args=(test_list, num_layers, model, Interpolation_matrix, u_shift, phi_r, obs_data, device),
            render_distributions=True,
        )
    except Exception as e:
        print("[note] Skipping render_model:", e)

    pyro.set_rng_seed(2025)

    # prior predictive (for diagnostics / optional plotting)
    prior = Predictive(pyro_model, num_samples=1000)(
        test_list, num_layers, model, Interpolation_matrix, u_shift, phi_r, obs_data, device
    )
    data = az.from_pyro(prior=prior)
    ###############################################################################
    # Posterior via SVI
    ###############################################################################
    clear_param_store()
    primitives.enable_validation(is_validate=True)
    
    # 3) less noisy ELBO: use a few particles
    

    #guide = pyro.infer.autoguide.AutoNormal(pyro_model)  # AutoDiagonalNormal is also fine
    #optim = ClippedAdam({"lr": 1e-2})
    adam_params = {"lr":  1.0 * 1e-5, "betas": (0.95, 0.999)}
    optim = Adam(adam_params)
    #svi = SVI(pyro_model, guide, optim, loss=Trace_ELBO())
    svi = SVI(pyro_model, custom_guide, optim, loss=Trace_ELBO(num_particles=20))
    num_steps = int(os.getenv("SVI_STEPS", 90000))
    log_every = max(1, num_steps // 50)
    t0 = datetime.now()
    losses = []
    for step in range(1, num_steps + 1):
        loss = svi.step(test_list, num_layers, model, Interpolation_matrix, u_shift, phi_r, obs_data, device)
        losses.append(loss)
        if step % log_every == 0 or step == 1:
            print(f"[SVI] step {step:5d}/{num_steps} | ELBO: {loss:.3f}")
    print(f"[SVI] finished in {(datetime.now() - t0).total_seconds():0.2f}s")

    # draw samples from the variational posterior (guide)
    num_post = int(os.getenv("SVI_SAMPLES", 1000))
    #predictive = Predictive(pyro_model, guide=guide, num_samples=num_post, return_sites=None)
    predictive = Predictive(pyro_model, guide=custom_guide, num_samples=num_post, return_sites=None)
    with torch.no_grad():
        posterior_draws = predictive(test_list, num_layers, model, Interpolation_matrix, u_shift, phi_r, obs_data, device)

    # collect latent names (mu_1, mu_2, …)
    param_names = [k for k, v in posterior_draws.items() if k.startswith("mu_") and v.ndim >= 1]

    # summarize parameters and save to CSV
    list_parameter_mean = []
    list_parameter_mean_plus_std = []
    list_parameter_mean_minus_std = []
    summary_stats = {}

    for i, name in enumerate(sorted(param_names, key=lambda x: int(x.split("_")[-1]))):
        print(i, name)
        vals = posterior_draws[name]  # [num_samples, ...]
        vals = vals.reshape(vals.shape[0], -1)
        m = vals.mean(0).mean()
        s = vals.std(0, unbiased=True).mean()

        print("Prior mean: ", test_list[i]["normal"]["mean"], "Prior std: ", test_list[i]["normal"]["std"])
        print("Posterior mean: ", m, "Posterior std: ", s)

        summary_stats[f"{name}_mean"] = float(m)
        summary_stats[f"{name}_std"] = float(s)

        list_parameter_mean.append(m.to(torch.float64))
        list_parameter_mean_plus_std.append((m + s).to(torch.float64))
        list_parameter_mean_minus_std.append((m - s).to(torch.float64))
    
    # add observation metadata for later grouping/summaries
    summary_stats["obs_count"] = int(targets.shape[0])
    
    uniq = np.unique(targets[:, [0, 1]], axis=0)  # compact (x,z) pairs
    stride = max(1, len(uniq) // 20)              # at most ~20 entries for brevity
    summary_stats["obs_coords"] = ";".join(f"({x:.2f},{z:.2f})" for x, z in uniq[::stride])

    summary_stats["elbo_final"] = float(loss)
    pd.DataFrame.from_dict(summary_stats, orient="index").to_csv("pyro_summary.csv")

    # optional: simple trace/density plots from draws (SVI)
    try:
        #az_data = az.from_dict(posterior={n: posterior_draws[n].cpu().numpy() for n in param_names})
        az_data = az.from_dict(
    posterior={n: posterior_draws[n].cpu().numpy()[None, :] for n in param_names}
    )
        plt.figure(figsize=(8, 10))
        az.plot_trace(az_data, var_names=param_names)
        plt.tight_layout()
        plt.savefig("svi_trace.png", dpi=300)   # ✅ save figure
        plt.close()                             # optional: close to free memory

        print("[SVI] Trace plot saved as svi_trace.png")
    except Exception as e:
        print("[SVI] Skipping ArviZ trace plot:", e)

    try:
        for i, name in enumerate(sorted(param_names, key=lambda x: int(x.split("_")[-1]))):
            plt.figure(figsize=(8, 10))
            az.plot_density(
                data=[{name: posterior_draws[name].cpu().numpy()}, data.prior],
                shade=0.9, bw=0.005, var_names=[name], data_labels=["Posterior (SVI)", "Prior"]
            )
            os.makedirs("./saved_model", exist_ok=True)
            plt.savefig(f"./saved_model/{name}.png")
            plt.close()
    except Exception as e:
        print("[SVI] Skipping density plots:", e)

    # final GemPy model renders (mean / mean±std)
    generate_final_model(
        geo_model=gempy_model, interpolation_input_=test_list, num_layers=num_layers,
        posterior_data=list_parameter_mean, slope=200, filename="posterior_model.png"
    )
    generate_final_model(
        geo_model=gempy_model, interpolation_input_=test_list, num_layers=num_layers,
        posterior_data=list_parameter_mean_plus_std, slope=200, filename="posterior_model_plus_std.png"
    )
    generate_final_model(
        geo_model=gempy_model, interpolation_input_=test_list, num_layers=num_layers,
        posterior_data=list_parameter_mean_minus_std, slope=200, filename="posterior_model_minus_std.png"
    )


if __name__ == "__main__":
    print("Script started...")
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print("Script ended...")
    print(f"Elapsed time: {end_time - start_time}")
