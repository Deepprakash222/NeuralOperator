import torch
import numpy as np
from torch.autograd import grad
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive

from pyro.nn import PyroModule, PyroSample

import gempy as gp
import gempy_engine
from gempy.core.data import Grid
from gempy_engine.core.backend_tensor import BackendTensor

import dolfin as dl

from datetime import datetime

from helpers import *
import json


class GempyModel(PyroModule):
    def __init__(self, interpolation_input_, geo_model_test, num_layers, slope, dtype, device, seed=None):
        super(GempyModel, self).__init__()
        BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)
        self.gempy_engine = gempy_engine
        self.interpolation_input_ = interpolation_input_
        self.geo_model_test = geo_model_test
        self.num_layers = num_layers
        self.dtype = dtype
        self.geo_model_test.interpolation_options.sigmoid_slope = slope
        self.device = device

        # ---------------- Seeding (fixed via argument from driver) ----------------
        if seed is None:
            seed = 42  # fallback default (driver should pass an explicit value)
        np.random.seed(seed)
        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def create_sample(self):
        """
        Define Pyro priors over the interpolation_input_ parameters.
        """
        Random_variable = {}
        counter = 1
        for interpolation_input_data in self.interpolation_input_[:self.num_layers]:
            if interpolation_input_data["update"] == "interface_data":
                if interpolation_input_data["prior_distribution"] == "normal":
                    mean = interpolation_input_data["normal"]["mean"]
                    std = interpolation_input_data["normal"]["std"]
                    Random_variable["mu_" + str(counter)] = pyro.sample("mu_" + str(counter), dist.Normal(mean, std))
                elif interpolation_input_data["prior_distribution"] == "uniform":
                    vmin = interpolation_input_data["uniform"]["min"]
                    vmax = interpolation_input_data["uniform"]["max"]
                    Random_variable["mu_" + str(counter)] = pyro.sample("mu_" + str(counter), dist.Uniform(vmin, vmax))
                else:
                    print("We have to include the distribution")
            counter = counter + 1

    def GenerateInputSamples(self, number_samples):
        pyro.clear_param_store()
        num_samples = int(number_samples)
        predictive = Predictive(self.create_sample, num_samples=num_samples)
        samples = predictive()

        samples_list = []
        for i in range(len(self.interpolation_input_)):
            samples_list.append(samples["mu_" + str(i + 1)].reshape(-1, 1))
        parameters = torch.hstack(samples_list)  # (N, p)

        return parameters.cpu().detach().numpy()

    def GempyForward(self, *params):
        index = 0
        interpolation_input = self.geo_model_test.interpolation_input

        for interpolation_input_data in self.interpolation_input_[:self.num_layers]:
            if interpolation_input_data["direction"] == "X":
                interpolation_input.surface_points.sp_coords = torch.index_put(
                    interpolation_input.surface_points.sp_coords,
                    (torch.tensor([interpolation_input_data["id"]]), torch.tensor([0])),
                    params[index])
            elif interpolation_input_data["direction"] == "Y":
                interpolation_input.surface_points.sp_coords = torch.index_put(
                    interpolation_input.surface_points.sp_coords,
                    (torch.tensor([interpolation_input_data["id"]]), torch.tensor([1])),
                    params[index])
            elif interpolation_input_data["direction"] == "Z":
                interpolation_input.surface_points.sp_coords = torch.index_put(
                    interpolation_input.surface_points.sp_coords,
                    (interpolation_input_data["id"], torch.tensor([2])),
                    params[index])
            else:
                print("Wrong direction")
            index = index + 1

        self.geo_model_test.solutions = self.gempy_engine.compute_model(
            interpolation_input=interpolation_input,
            options=self.geo_model_test.interpolation_options,
            data_descriptor=self.geo_model_test.input_data_descriptor,
            geophysics_input=self.geo_model_test.geophysics_input,
        )

        m_samples = self.geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
        return m_samples

    def GenerateOutputSamples_(self, Inputs_samples):
        from torch.autograd.functional import jacobian

        Inputs_samples = torch.tensor(Inputs_samples, dtype=self.dtype, device=self.device)
        m_data = []
        dmdc_data = []
        for i in range(Inputs_samples.shape[0]):
            params_tuple = tuple([Inputs_samples[i, j].clone().requires_grad_(True) for j in range(Inputs_samples.shape[1])])

            m_samples = self.GempyForward(*params_tuple)
            m_data.append(m_samples.detach())

            J = jacobian(self.GempyForward, params_tuple)
            J_matrix = torch.stack(J, dim=1)
            dmdc_data.append(J_matrix.detach())

        return torch.stack(m_data).numpy(), torch.stack(dmdc_data).numpy()


def generate_input_output_gempy_data(mesh, nodes, number_samples, comm, device, slope=200, filename=None, seed=None):
    mesh_coordinates = mesh.coordinates()
    global_indices = mesh.topology().global_indices(0)  # vertex global IDs

    data = {}
    geo_model_test = create_initial_gempy_model(refinement=3, save=False)

    if mesh_coordinates.shape[1] == 2:
        xyz_coord = np.insert(mesh_coordinates, 1, 0, axis=1)
    elif mesh_coordinates.shape[1] == 3:
        xyz_coord = mesh_coordinates
    else:
        raise ValueError("Mesh coordinates must be 2D or 3D.")

    gp.set_custom_grid(geo_model_test.grid, xyz_coord=xyz_coord)
    geo_model_test.interpolation_options.mesh_extraction = False

    sp_coords_copy_test = geo_model_test.interpolation_input.surface_points.sp_coords.copy()

    # Store initial data only on rank 0 to avoid race conditions
    if comm.Get_rank() == 0:
        df_sp_init = geo_model_test.surface_points.df
        df_or_init = geo_model_test.orientations.df
        df_sp_init.to_csv("./Initial_sp.csv")
        df_or_init.to_csv("./Initial_op.csv")

    # Random variable setup
    dtype = torch.float64
    test_list = []
    std = 0.03  # 0.125 , 4*std = gap between two layers

    vmin = torch.tensor(sp_coords_copy_test[1, 2], dtype=dtype, device=device) - torch.tensor(std, dtype=dtype, device=device)
    vmax = torch.tensor(sp_coords_copy_test[1, 2], dtype=dtype, device=device) + torch.tensor(std, dtype=dtype, device=device)
    test_list.append({
        "update": "interface_data",
        "id": torch.tensor([1]),
        "direction": "Z",
        "prior_distribution": "uniform",
        "uniform": {"min": vmin, "max": vmax}
    })

    num_layers = len(test_list)

    # Use the seed passed from the driver
    model_seed = seed if seed is not None else 42
    Gempy = GempyModel(test_list, geo_model_test, num_layers, slope=slope, dtype=dtype, device=device, seed=model_seed)

    comm.Barrier()

    # Generate inputs on rank 0
    if comm.Get_rank() == 0:
        c = Gempy.GenerateInputSamples(number_samples=number_samples)
        data["input"] = c.tolist()
    else:
        c = None

    # Broadcast inputs to all ranks
    c = comm.bcast(c, root=0)

    # Compute outputs & Jacobians
    m_data, dmdc_data = Gempy.GenerateOutputSamples_(Inputs_samples=c)

    # Local results: (global_vertex_index, outputs_for_all_samples_at_this_vertex, jacobians_at_this_vertex)
    local_results = [(int(global_indices[idx]), m_data[:, idx], dmdc_data[:, idx]) for idx in range(global_indices.shape[0])]
    comm.Barrier()

    # Gather all ranks to rank 0
    all_results = comm.gather(local_results, root=0)

    if comm.Get_rank() == 0:
        global_output = np.zeros((c.shape[0], nodes))
        global_gradient = np.zeros((c.shape[0], nodes, num_layers))
        for result_list in all_results:
            for idx_, output_, grad_ in result_list:
                global_output[:, idx_] = output_
                global_gradient[:, idx_] = grad_
        data["Gempy_output"] = global_output.tolist()
        data["Jacobian_Gempy"] = global_gradient.tolist()

    return data


def create_true_data(mesh, nodes, slope=200, filename=None):
    mesh_coordinates = mesh.coordinates()
    geo_model_test = create_initial_gempy_model(refinement=7, save=True)

    if mesh_coordinates.shape[1] == 2:
        xyz_coord = np.insert(mesh_coordinates, 1, 0, axis=1)
    elif mesh_coordinates.shape[1] == 3:
        xyz_coord = mesh_coordinates
    else:
        raise ValueError("Mesh coordinates must be 2D or 3D.")

    gp.set_custom_grid(geo_model_test.grid, xyz_coord=xyz_coord)
    geo_model_test.interpolation_options.mesh_extraction = False
    sol = gp.compute_model(geo_model_test)

    geo_model_test.interpolation_options.sigmoid_slope = slope
    gp.compute_model(geo_model_test)
    sp_coords_copy_test = geo_model_test.interpolation_input.surface_points.sp_coords.copy()
    m_initial = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values

    return m_initial, sp_coords_copy_test, geo_model_test


def generate_final_model(geo_model, interpolation_input_, num_layers, posterior_data, slope=200, filename='posterior_model.png', save=True):
    BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)

    interpolation_input = geo_model.interpolation_input
    geo_model.interpolation_options.sigmoid_slope = slope
    index = 0
    for interpolation_input_data in interpolation_input_[:num_layers]:
        if interpolation_input_data["direction"] == "X":
            interpolation_input.surface_points.sp_coords = torch.index_put(
                interpolation_input.surface_points.sp_coords,
                (torch.tensor([interpolation_input_data["id"]]), torch.tensor([0])),
                posterior_data[index])
        elif interpolation_input_data["direction"] == "Y":
            interpolation_input.surface_points.sp_coords = torch.index_put(
                interpolation_input.surface_points.sp_coords,
                (torch.tensor([interpolation_input_data["id"]]), torch.tensor([1])),
                posterior_data[index])
        elif interpolation_input_data["direction"] == "Z":
            interpolation_input.surface_points.sp_coords = torch.index_put(
                interpolation_input.surface_points.sp_coords,
                (interpolation_input_data["id"], torch.tensor([2])),
                posterior_data[index])
        else:
            print("Wrong direction")
        index = index + 1

    Phyiscal_Data = geo_model.transform.apply_inverse(interpolation_input.surface_points.sp_coords.detach().numpy())
    posterior_data_ = [Phyiscal_Data[1, 2]]
    print(Phyiscal_Data)
    geo_model_test = create_final_gempy_model(posterior_data_, refinement=7, save=save, filename=filename)
    geo_model_test.interpolation_options.mesh_extraction = False
    sol = gp.compute_model(geo_model_test)
    geo_model_test.interpolation_options.sigmoid_slope = slope
    gp.compute_model(geo_model_test)


def generate_input_output_gempy_data_(mesh, nodes, number_samples, device, slope=200, filename=None, seed=None):
    mesh_coordinates = mesh.coordinates()

    data = {}
    geo_model_test = create_initial_gempy_model(refinement=3, save=True)
    if mesh_coordinates.shape[1] == 2:
        xyz_coord = np.insert(mesh_coordinates, 1, 0, axis=1)
    elif mesh_coordinates.shape[1] == 3:
        xyz_coord = mesh_coordinates
    else:
        raise ValueError("Mesh coordinates must be 2D or 3D.")

    gp.set_custom_grid(geo_model_test.grid, xyz_coord=xyz_coord)
    geo_model_test.interpolation_options.mesh_extraction = False

    sp_coords_copy_test = geo_model_test.interpolation_input.surface_points.sp_coords.copy()

    # Store initial (non-MPI)
    df_sp_init = geo_model_test.surface_points.df
    df_or_init = geo_model_test.orientations.df
    df_sp_init.to_csv("./Initial_sp.csv")
    df_or_init.to_csv("./Initial_op.csv")

    dtype = torch.float64
    test_list = []
    std = 0.03
    test_list.append({
        "update": "interface_data",
        "id": torch.tensor([1]),
        "direction": "Z",
        "prior_distribution": "normal",
        "normal": {"mean": torch.tensor(sp_coords_copy_test[1, 2], dtype=dtype), "std": torch.tensor(std, dtype=dtype)}
    })
    num_layers = len(test_list)

    model_seed = seed if seed is not None else 42
    Gempy = GempyModel(test_list, geo_model_test, num_layers, slope=slope, dtype=dtype, device=device, seed=model_seed)

    c = Gempy.GenerateInputSamples(number_samples=number_samples)
    m_data, dmdc_data = Gempy.GenerateOutputSamples_(Inputs_samples=c)

    data["input"] = c
    data["Gempy_output"] = m_data
    data["Jacobian_Gempy"] = dmdc_data
    return data
