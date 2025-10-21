# ... your existing imports ...
import argparse
import os
from datetime import datetime
import numpy as np
import torch
import dolfin as dl
from mpi4py import MPI
import json

from helpers import *
from generate_samples import generate_input_output_gempy_data  # (same name as before)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10, help="Number of input samples to generate (default: 10)")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for reproducibility (default: None = time-based)")
    args = parser.parse_args()

    # Device selection
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # device = torch.device("mps")
        device = torch.device("cpu")  # keep CPU if you prefer; switch to mps if desired
        print("Using MPS device (overridden to CPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Process {rank} of {size}")

    # Mesh
    nx = 31
    ny = 31
    mesh = dl.UnitSquareMesh(comm, nx, ny)
    nodes = (nx + 1) * (ny + 1)

    # Output dir (rank 0)
    if rank == 0:
        directory_path = f"../Results/Nodes_{nodes}"
        os.makedirs(directory_path, exist_ok=True)
        print(f"Output dir: {directory_path}")
    else:
        directory_path = None

    comm.Barrier()

    # ---- RUN with configurable samples + seed ----
    data = generate_input_output_gempy_data(
        mesh=mesh,
        nodes=nodes,
        number_samples=args.num_samples,   # <<<<<<<<<< only 10 by default
        comm=comm,
        device=device,
        seed=args.seed                     # <<<<<<<<<< configurable seed (None => time-based)
    )

    comm.Barrier()

    # Save on rank 0
    if rank == 0:
        filename = os.path.join(directory_path, "data_1_parameter_uniform.json")
        c, m_data, dmdc_data = np.array(data["input"]), np.array(data["Gempy_output"]), np.array(data["Jacobian_Gempy"])
        print("Shapes-", "Gempy Input: ", c.shape, "Gempy Output:", m_data.shape, "Jacobian shape:", dmdc_data.shape)
        with open(filename, "w") as f:
            json.dump(data, f)
        print(f"Wrote: {filename}")


if __name__ == "__main__":
    print("Script started...")
    start_time = datetime.now()

    main()

    end_time = datetime.now()
    print("Script ended...")
    print(f"Elapsed time: {end_time - start_time}")
