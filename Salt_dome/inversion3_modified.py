# inversion3.py — SVI version (device-safe, plots, tidy CSV, final model)

# (optional) lock to a single GPU to silence fork_rng warnings:
# import os as _os; _os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import os
import json
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
from pyro.optim import ClippedAdam
from pyro.infer.autoguide import AutoNormal


# ---------------------------
# Pyro model (device-aware)
# ---------------------------
def pyro_model(interpolation_input_, num_layers, NN_model, Interpolation_matrix, u_shift, phi, obs_data, device):
    """
    Priors over mu_i, NN->field->interpolate, Gaussian likelihood with learned sigma.
    """
    params = []
    counter = 1
    for d in interpolation_input_[:num_layers]:
        if d["update"] != "interface_data":
            counter += 1
            continue

        if d["prior_distribution"] == "normal":
            mean_t = d["normal"]["mean"]
            std_t  = d["normal"]["std"]
            mean_t = mean_t.to(device) if isinstance(mean_t, torch.Tensor) else torch.tensor(mean_t, device=device)
            std_t  = std_t.to(device)  if isinstance(std_t, torch.Tensor)  else torch.tensor(std_t, device=device)
            params.append(pyro.sample(f"mu_{counter}", dist.Normal(mean_t, std_t)))
        elif d["prior_distribution"] == "uniform":
            mn = d["uniform"]["min"]
            mx = d["uniform"]["max"]
            mn = mn.to(device) if isinstance(mn, torch.Tensor) else torch.tensor(mn, device=device)
            mx = mx.to(device) if isinstance(mx, torch.Tensor) else torch.tensor(mx, device=device)
            params.append(pyro.sample(f"mu_{counter}", dist.Uniform(mn, mx)))
        else:
            raise ValueError("Unsupported prior_distribution")
        counter += 1

    input_data = torch.stack(params).to(device) if params else torch.empty((0,), dtype=torch.float32, device=device)

    # keep grads (no torch.no_grad)
    NN_output = NN_model(input_data)                  # CHANGED: allow grad
    output = torch.matmul(Interpolation_matrix, torch.matmul(NN_output, phi.T) + u_shift)

    # CHANGED: device-aware sigma prior
    sigma = pyro.sample("sigma", dist.HalfCauchy(torch.tensor(0.1, device=device)))

    with pyro.plate("obs_plate", obs_data.shape[0]):
        pyro.sample("obs", dist.Normal(output, sigma), obs=obs_data)


def main():
    # ---------- device ----------
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("cpu")
        print("MPS available — using CPU for stability")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    # ---------- FEM / PDE setup ----------
    nx = 31; ny = 31
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

    def pde_varf(u,m,p):
        return m*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx - f*p*ufl.dx

    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

    # ---------- observation locations ----------
    xs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    Borehole_extent = [0.1]*len(xs)
    Borehole_points = [20]*len(xs)
    tgt = []
    for i,x in enumerate(xs):
        z = np.linspace(Borehole_extent[i], 0.9, Borehole_points[i])
        tgt.append(np.column_stack((np.full_like(z, x), z)))
    targets = np.vstack(tgt)
    B = hp.assemblePointwiseObservation(Vh[hp.STATE], targets)

    # ---------- truth + noisy obs ----------
    u = dl.Function(Vh[hp.STATE]); uadj = dl.Function(Vh[hp.ADJOINT])
    m = dl.Function(Vh[hp.PARAMETER])

    m_initial, sp_coords_copy_test, gempy_model = create_true_data(mesh=mesh, nodes=nodes, filename=None)
    m_initial = 2 * m_initial
    m.vector().set_local(m_initial[d2v])

    x = [u.vector(), m.vector(), uadj.vector()]
    pde.solveFwd(x[hp.STATE], x)

    observable_2 = hf.LinearStateObservable(pde, B)
    u_obs = observable_2.evalu(x[hp.STATE]).get_local()  # numpy

    # ---------- reduced-order + NN ----------
    dtype = torch.float32
    Mphi_r = torch.tensor(np.load('./saved_model/Mphi_r.npy'), dtype=dtype, device=device)
    phi_r   = torch.tensor(np.load('./saved_model/phi_r.npy'), dtype=dtype, device=device)
    u_shift = torch.tensor(np.load('./saved_model/u_shift.npy'), dtype=dtype, device=device)

    # load & MOVE model to device
    model_jacobian_full = torch.load("./saved_model/model_jacobian_full.pth", map_location=device, weights_only=False)
    model_jacobian_full.to(device)                    # CHANGED
    model_jacobian_full.eval()

    # ---------- priors ----------
    test_list=[]
    std = 0.03
    test_list.append({
        "update":"interface_data","id":torch.tensor([1]),
        "direction":"Z",
        "prior_distribution":"normal",
        "normal":{
            "mean": torch.tensor(sp_coords_copy_test[1,2], dtype=dtype, device=device),
            "std":  torch.tensor(std, dtype=dtype, device=device)
        }
    })

    num_layers = len(test_list)
    model = model_jacobian_full
    Interpolation_matrix = torch.tensor(B.array(), dtype=dtype, device=device)
    obs_data = torch.tensor(u_obs, dtype=dtype, device=device)   # CHANGED: device tensor

    # (optional) render
    try:
        render_model(
            pyro_model,
            model_args=(test_list, num_layers, model, Interpolation_matrix, u_shift, phi_r, obs_data, device),
            render_distributions=True
        )
    except Exception as e:
        print("[note] render_model skipped:", e)

    pyro.set_rng_seed(42)

    # ---------- SVI inference ----------
    clear_param_store()
    primitives.enable_validation(is_validate=True)
    guide = AutoNormal(pyro_model)
    optim = ClippedAdam({"lr": 1e-3})
    svi = SVI(pyro_model, guide, optim, loss=Trace_ELBO())

    steps = int(os.getenv("SVI_STEPS", 3000))
    log_every = max(1, steps // 50)
    print("Running SVI...")
    t0 = datetime.now()
    for s in range(1, steps+1):
        loss = svi.step(test_list, num_layers, model, Interpolation_matrix, u_shift, phi_r, obs_data, device)
        if s % log_every == 0 or s == 1:
            print(f"[SVI] step {s:5d}/{steps} | ELBO: {loss:.3f}")
    print(f"[SVI] finished in {(datetime.now()-t0).total_seconds():.2f}s")

    # ---------- posterior draws ----------
    draws = int(os.getenv("SVI_SAMPLES", 1000))
    predictive = Predictive(pyro_model, guide=guide, num_samples=draws, return_sites=None)
    with torch.no_grad():
        post = predictive(test_list, num_layers, model, Interpolation_matrix, u_shift, phi_r, obs_data, device)

    names = [k for k in post if k.startswith("mu_")] + (["sigma"] if "sigma" in post else [])

    # summarize mu_* (for final models)
    mu_names = sorted([n for n in post if n.startswith("mu_")], key=lambda x: int(x.split("_")[-1]))
    list_parameter_mean, list_parameter_mean_plus_std, list_parameter_mean_minus_std = [], [], []
    summary_stats = {}

    for i, name in enumerate(mu_names):
        vals = post[name].reshape(post[name].shape[0], -1)   # (draws, ...)
        m = vals.mean(0).mean()
        s = vals.std(0, unbiased=True).mean()
        print("Prior mean:", test_list[i]["normal"]["mean"], "Prior std:", test_list[i]["normal"]["std"])
        print("Posterior mean:", m, "Posterior std:", s)
        summary_stats[f"{name}_mean"] = float(m)
        summary_stats[f"{name}_std"]  = float(s)
        list_parameter_mean.append(float(m))
        list_parameter_mean_plus_std.append(float(m + s))
        list_parameter_mean_minus_std.append(float(m - s))

    if "sigma" in post:
        svals = post["sigma"].reshape(post["sigma"].shape[0], -1)
        summary_stats["sigma_mean"] = float(svals.mean())
        summary_stats["sigma_std"]  = float(svals.std())

    summary_stats["obs_count"] = int(targets.shape[0])
    uniq = np.unique(targets[:, [0, 1]], axis=0)
    stride = max(1, len(uniq)//20)
    summary_stats["obs_coords"] = ";".join(f"({x:.2f},{z:.2f})" for x,z in uniq[::stride])
    summary_stats["elbo_final"] = float(loss)
    pd.DataFrame.from_dict(summary_stats, orient="index").to_csv("pyro_summary.csv")

    # tidy CSV of posterior draws (one row per draw)
    tidy_rows = []
    for d in range(draws):
        row = {"draw": d}
        for name in names:
            arr = post[name][d].detach().cpu().numpy().ravel()
            row[name] = float(arr.item()) if arr.size == 1 else json.dumps(arr.tolist())
        tidy_rows.append(row)
    pd.DataFrame(tidy_rows).to_csv("pyro_summary_tidy.csv", index=False)
    print(f"Wrote {len(tidy_rows)} posterior rows to pyro_summary_tidy.csv (header written=True)")

    # ---------- ArviZ trace ----------
    try:
        az_posterior = {}
        for name in names:
            arr = post[name].detach().cpu().numpy()  # (draws, ...)
            if arr.ndim == 1:
                arr = arr[None, :]                  # (1, draws)
            else:
                arr = arr[None, ...]                # (1, draws, ...)
            az_posterior[name] = arr
        az_data = az.from_dict(posterior=az_posterior)
        os.makedirs("./saved_model", exist_ok=True)
        plt.figure(figsize=(8,10))
        az.plot_trace(az_data, var_names=list(az_posterior.keys()))
        plt.savefig("./saved_model/svi_trace.png", bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("ArviZ plotting skipped:", e)

    # density plots
    try:
        os.makedirs("./saved_model", exist_ok=True)
        for name in names:
            plt.figure(figsize=(6,6))
            data_dict = {name: post[name].detach().cpu().numpy()}
            az.plot_density(data=[data_dict], var_names=[name], shade=0.9)
            plt.title(name)
            plt.savefig(f"./saved_model/{name}.png", bbox_inches="tight")
            plt.close()
    except Exception as e:
        print("Density plots skipped:", e)

    # ---------- build final GemPy models ----------
    try:
        # CHANGED: pass CPU tensors of floats to helper
        posterior_mean_t  = torch.tensor(list_parameter_mean, dtype=torch.float32, device="cpu")
        posterior_plus_t  = torch.tensor(list_parameter_mean_plus_std, dtype=torch.float32, device="cpu")
        posterior_minus_t = torch.tensor(list_parameter_mean_minus_std, dtype=torch.float32, device="cpu")

        generate_final_model(
            geo_model=gempy_model, interpolation_input_=test_list, num_layers=num_layers,
            posterior_data=posterior_mean_t, slope=200, filename='posterior_model.png'
        )
        generate_final_model(
            geo_model=gempy_model, interpolation_input_=test_list, num_layers=num_layers,
            posterior_data=posterior_plus_t, slope=200, filename='posterior_model_plus_std.png'
        )
        generate_final_model(
            geo_model=gempy_model, interpolation_input_=test_list, num_layers=num_layers,
            posterior_data=posterior_minus_t, slope=200, filename='posterior_model_minus_std.png'
        )
    except Exception as e:
        print("generate_final_model skipped or failed:", e)

    print("Done inference run.")


if __name__ == "__main__":
    print("Script started.")
    start = datetime.now()
    main()
    end = datetime.now()
    print("Script ended.")
    print("Elapsed:", end - start)
