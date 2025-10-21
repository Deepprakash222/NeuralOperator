import os, sys
from datetime import datetime
import json
import numpy as np

import torch
import gempy as gp
import gempy_engine
import gempy_viewer as gpv
import dolfin as dl

from helpers import *
from generate_samples import generate_input_output_gempy_data
from mpi4py import MPI

import warnings
warnings.filterwarnings("ignore")


def main():
    # ---- Device selection ----
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # device = torch.device("mps")
        device = torch.device("cpu")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    # ---- MPI ----
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Process {rank} of {size}")

    # ---- Mesh ----
    nx = 31
    ny = 31
    mesh = dl.UnitSquareMesh(comm, nx, ny)

    nodes = (nx + 1) * (ny + 1)
    if rank == 0:
        directory_path = f"../Results/Nodes_{nodes}"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' was created.")
        else:
            print(f"Directory '{directory_path}' already exists.")
    else:
        directory_path = None

    comm.Barrier()

    # ---- Fixed seed set here (no console args needed) ----
    SEED_VALUE = 1234

    # ---- Use only 10 samples for a quick test ----
    data = generate_input_output_gempy_data(
        mesh=mesh,
        nodes=nodes,
        number_samples=10,         # << test with 10 samples
        comm=comm,
        device=device,
        seed=SEED_VALUE            # << fixed seed controlled here
    )

    comm.Barrier()
    if rank == 0:
        filename = os.path.join(directory_path, "data_1_parameter_uniform.json")
        c = np.array(data["input"])
        m_data = np.array(data["Gempy_output"])
        dmdc_data = np.array(data["Jacobian_Gempy"])
        print("Shapes-", "Gempy Input: ", c.shape, "Gempy Output:", m_data.shape, "Jacobian shape:", dmdc_data.shape)
        with open(filename, 'w') as file:
            json.dump(data, file)
        print(f"Wrote: {filename}")


if __name__ == "__main__":
    print("Script started...")
    start_time = datetime.now()

    main()

    end_time = datetime.now()
    print("Script ended...")
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time}")
