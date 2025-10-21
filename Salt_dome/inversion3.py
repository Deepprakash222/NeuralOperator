import json
import ufl
import dolfin as dl
from datetime import datetime
import hippylib as hp
import hippyflow as hf

from helpers import *
from generate_samples import *
from train_nn import *  # your NeuralNet lives here
import arviz as az
from pyro.infer import Predictive
from pyro.infer.autoguide import init_to_mean
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
import sys
from torch.serialization import add_safe_globals, safe_globals

# =========================
# Overlap controls (tunable)
# =========================
PRIOR_STD = 0.03          # broader prior so it overlaps with posterior
LIKELIHOOD_SIGMA = 0.05   # weaker likelihood -> posterior stays closer to prior

# -----------------------------------------------------------------------------
# Make checkpoints saved as "train_nn2.NeuralNet" resolvable + allowlisted
# -----------------------------------------------------------------------------
import train_nn as train_nn2
sys.modules['train_nn2'] = train_nn2
from train_nn2 import NeuralNet
add_safe_globals([NeuralNet])  # allowlist for weights_only=True in PyTorch 2.6


# -----------------------------------------------------------------------------
# Pyro model (SVI-compatible). NOTE: keep autograd through NN (no torch.no_grad()).
# -----------------------------------------------------------------------------
def pyro_model(interpolation_input_, num_layers, NN_model, Interpolation_matrix, u_shift, phi,
               obs_data, device, likelihood_sigma):
    """
    Probabilistic geological model in Pyro.
    Defines priors for interface parameters and observation likelihood.
    """
    parameter = []
    counter = 1
    for interpolation_input_data in interpolation_input_[:num_layers]:
        if interpolation_input_data["update"] == "interface_data":
            if interpolation_input_data["prior_distribution"] == "normal":
                mean = interpolation_input_data["normal"]["mean"]
                std = interpolation_input_data["normal"]["std"]
                rv = pyro.sample(f"mu_{counter}", dist.Normal(mean, std))
                parameter.append(rv.to(device))
            elif interpolation_input_data["prior_distribution"] == "uniform":
                min_ = interpolation_input_data["uniform"]["min"]
                max_ = interpolation_input_data["uniform"]["max"]
                rv = pyro.sample(f"mu_{counter}", dist.Uniform(min_, max_))
                parameter.append(rv.to(device))
            else:
                raise ValueError("Unsupported prior_distribution; use 'normal' or 'uniform'.")
        counter += 1

    # Stack into shape [num_layers]
    input_data = torch.stack(parameter).to(device)

    # Forward through surrogate model
    NN_model.eval()
    NN_output = NN_model(input_data)  # if your NN expects [B,F], use input_data.unsqueeze(0)

    # Project to observation space
    output = torch.matmul(Interpolation_matrix, torch.matmul(NN_output, phi.T) + u_shift)

    # Likelihood with configurable sigma
    with pyro.plate("likelihood", obs_data.shape[0]):
        pyro.sample("obs", dist.Normal(output, likelihood_sigma), obs=obs_data)


# -----------------------------------------------------------------------------
# Optional: robust helper for ArviZ conversion
# -----------------------------------------------------------------------------
def to_arviz_safe(posterior_samples, prior, posterior_predictive):
    try:
        return az.from_pyro(
            posterior=posterior_samples,
            prior=prior,
            posterior_predictive=posterior_predictive
        )
    except Exception:
        post_np = {k: v.detach().cpu().numpy() for k, v in posterior_samples.items()}
        prior_np = {}
        if isinstance(prior, dict):
            for k, v in prior.items():
                if isinstance(v, torch.Tensor) and k.startswith("mu_"):
                    prior_np[k] = v.detach().cpu().numpy()
        ppc_np = {}
        if isinstance(posterior_predictive, dict) and "obs" in posterior_predictive:
            ppc_np["obs"] = posterior_predictive["obs"].detach().cpu().numpy()
        return az.from_dict(
            posterior=post_np,
            prior=prior_np if prior_np else None,
            posterior_predictive=ppc_np if ppc_np else None
        )


def main():
    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("cpu")  # Fenics + MPS can be problematic
        print("Using MPS device (forcing CPU for compatibility)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    # -------------------------------------------------------------------------
    # FEniCS / PDE setup
    # -------------------------------------------------------------------------
    nx = 31
    ny = 31
    nodes = (nx + 1) * (ny + 1)
    mesh = dl.RectangleMesh(dl.Point(0.0, 0.0), dl.Point(1.0, 1.0), nx, ny)
    Vh_STATE = dl.FunctionSpace(mesh, "CG", 2)
    Vh_PARAMETER = dl.FunctionSpace(mesh, "CG", 1)
    Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE]

    d2v = dl.dof_to_vertex_map(Vh[hp.PARAMETER])

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

    # Observation operator (boreholes)
    Borehole_location = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    Borehole_extent = [0.1] * 9
    Borehole_points = [20] * 9
    target_list = []
    for i, x in enumerate(Borehole_location):
        z_data = np.linspace(Borehole_extent[i], 0.9, Borehole_points[i])
        first_column = np.full((z_data.shape[0],), x)
        two_d_array = np.column_stack((first_column, z_data))
        target_list.append(two_d_array)
    targets = np.vstack(target_list)
    B = hp.assemblePointwiseObservation(Vh[hp.STATE], targets)

    # True parameter field & state
    m_initial, sp_coords_copy_test, gempy_model = create_true_data(mesh=mesh, nodes=nodes, filename=None)
    m_initial = 2 * m_initial
    m = dl.Function(Vh[hp.PARAMETER])
    m.vector().set_local(m_initial[d2v])

    # Optional visualization of parameter field
    fig, ax = plt.subplots(figsize=(6, 6))
    im = plt.imshow(m_initial.reshape((nx + 1, nx + 1)), cmap='viridis', origin='lower',
                    interpolation='bilinear', extent=[0, 1, 0, 1])
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("x-coordinate")
    ax.set_ylabel("z-coordinate")
    ax.set_title("Parameter Field $m(x,z)$")
    plt.tight_layout()
    plt.show()

    u = dl.Function(Vh[hp.STATE])
    uadj = dl.Function(Vh[hp.ADJOINT])
    x = [u.vector(), m.vector(), uadj.vector()]
    pde.solveFwd(x[hp.STATE], x)
    u_true = x[hp.STATE].get_local()
    u_obs = B.array() @ u_true

    # Add Gaussian noise to observations (match the likelihood sigma)
    u_obs = u_obs + np.random.normal(loc=0.0, scale=LIKELIHOOD_SIGMA, size=u_obs.shape)

    # -------------------------------------------------------------------------
    # Load reduced-order items + NN model
    # -------------------------------------------------------------------------
    dtype = torch.float32
    Mphi_r = torch.tensor(np.load('./saved_model/Mphi_r.npy'), dtype=dtype, device=device)  # (loaded but unused)
    phi_r = torch.tensor(np.load('./saved_model/phi_r.npy'), dtype=dtype, device=device)
    u_shift = torch.tensor(np.load('./saved_model/u_shift.npy'), dtype=dtype, device=device)

    # Robust checkpoint load (PyTorch 2.6)
    ckpt_path = "./saved_model/model_jacobian_full.pth"
    try:
        # Safe load with weights_only=True (default) and allowlisted class
        with safe_globals([NeuralNet]):
            obj = torch.load(ckpt_path, map_location=device)  # weights_only=True by default in 2.6
    except Exception as e:
        print("[warn] Safe load failed; falling back to weights_only=False (trusted checkpoint):", repr(e))
        # Use this ONLY if you trust the checkpoint file (your own artifact)
        obj = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Normalize to ready-to-use model
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        # pure state_dict
        model_jacobian_full = NeuralNet().to(device)  # <-- supply ctor args if your model needs them
        model_jacobian_full.load_state_dict(obj)
        model_jacobian_full.eval()
    elif isinstance(obj, dict) and "state_dict" in obj:
        model_jacobian_full = NeuralNet().to(device)  # <-- ctor args if needed
        model_jacobian_full.load_state_dict(obj["state_dict"])
        model_jacobian_full.eval()
    else:
        # full nn.Module
        model_jacobian_full = obj.to(device)
        model_jacobian_full.eval()

    # -------------------------------------------------------------------------
    # Priors over interface(s) — broadened to encourage overlap
    # -------------------------------------------------------------------------
    std = PRIOR_STD
    test_list = [{
        "update": "interface_data",
        "id": torch.tensor([1]),
        "direction": "Z",
        "prior_distribution": "normal",
        "normal": {
            "mean": torch.tensor(sp_coords_copy_test[1, 2], dtype=dtype, device=device),
            "std": torch.tensor(std, dtype=dtype, device=device)
        }
    }]
    num_layers = len(test_list)

    Interpolation_matrix = torch.tensor(B.array(), dtype=dtype, device=device)
    obs_data = torch.tensor(u_obs, dtype=dtype, device=device)
    model = model_jacobian_full

    # -------------------------------------------------------------------------
    # Prior predictive (latents)
    # -------------------------------------------------------------------------
    pyro.set_rng_seed(42)
    prior = Predictive(pyro_model, num_samples=1000)(
        test_list, num_layers, model, Interpolation_matrix, u_shift, phi_r, obs_data, device,
        LIKELIHOOD_SIGMA
    )

    # -------------------------------------------------------------------------
    # Posterior via SVI
    # -------------------------------------------------------------------------
    from pyro.infer import SVI, Trace_ELBO
    from pyro.infer.autoguide import AutoDiagonalNormal
    from pyro.optim import ClippedAdam

    guide = AutoDiagonalNormal(pyro_model, init_loc_fn=init_to_mean)
    svi = SVI(pyro_model, guide, ClippedAdam({"lr": 3e-3, "clip_norm": 5.0}), loss=Trace_ELBO())

    pyro.clear_param_store()
    num_steps = 4000
    for step in range(num_steps):
        loss = svi.step(
            test_list, num_layers, model, Interpolation_matrix, u_shift, phi_r, obs_data, device,
            LIKELIHOOD_SIGMA
        )
        if (step + 1) % 500 == 0:
            print(f"[SVI] step {step + 1}/{num_steps}, loss = {loss:.4f}")

    # Draw samples from the variational posterior
    num_posterior_samples = 2000
    posterior_samples = Predictive(guide, num_samples=num_posterior_samples)(
        test_list, num_layers, model, Interpolation_matrix, u_shift, phi_r, obs_data, device,
        LIKELIHOOD_SIGMA
    )
    # Keep only latent mu_* sites
    posterior_samples = {k: v for k, v in posterior_samples.items() if k.startswith("mu_")}

    # Summaries
    summary_stats = {k: {"mean": torch.mean(v).item(), "std": torch.std(v).item()} for k, v in posterior_samples.items()}
    pd.DataFrame(summary_stats).T.to_csv("pyro_summary.csv")

    list_parameter_mean = []
    list_parameter_mean_plus_std = []
    list_parameter_mean_minus_std = []
    for i in range(len(test_list)):
        name = f"mu_{i + 1}"
        vals = posterior_samples[name]
        mean_val = torch.mean(vals)
        std_val = torch.std(vals)
        print("Prior mean:", test_list[i]["normal"]["mean"], "Prior std:", test_list[i]["normal"]["std"])
        print("Posterior mean (SVI):", mean_val, "Posterior std (SVI):", std_val)
        list_parameter_mean.append(mean_val.to(torch.float64))
        list_parameter_mean_plus_std.append((mean_val + std_val).to(torch.float64))
        list_parameter_mean_minus_std.append((mean_val - std_val).to(torch.float64))

    # Posterior predictive
    posterior_predictive = Predictive(pyro_model, guide=guide, num_samples=1000)(
        test_list, num_layers, model, Interpolation_matrix, u_shift, phi_r, obs_data, device,
        LIKELIHOOD_SIGMA
    )

    # ArviZ conversion & plots (robust)
    data = to_arviz_safe(posterior_samples, prior, posterior_predictive)
    az.plot_trace(data)

    for i in range(len(test_list)):
        plt.figure(figsize=(8, 10))
        az.plot_density(
            data=[data.posterior, data.prior],
            shade=0.9,
            bw=0.03,
            var_names=[f"mu_{i + 1}"],
            data_labels=["Posterior (SVI)", "Prior"]
        )
        plt.savefig(f"./saved_model/mu_{i}.png")
        plt.close()

    # Reconstruct final models at mean and ±std
    generate_final_model(gempy_model, test_list, num_layers, list_parameter_mean, 200, 'posterior_model.png')
    generate_final_model(gempy_model, test_list, num_layers, list_parameter_mean_plus_std, 200, 'posterior_model_plus_std.png')
    generate_final_model(gempy_model, test_list, num_layers, list_parameter_mean_minus_std, 200, 'posterior_model_minus_std.png')


if __name__ == "__main__":
    print("Script started...")
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print("Script ended.")
    print(f"Elapsed time: {end_time - start_time}")
