import os,sys
from datetime import datetime
import numpy as np

import gempy as gp
import gempy_engine
import gempy_viewer as gpv

import dolfin as dl


from helpers import *
from generate_samples import *
from mpi4py import MPI

import warnings
warnings.filterwarnings("ignore")


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
    
    # Get the MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    
    print(f"Process {rank} of {size}")
    
    # ---------------- 1️⃣ Create the Mesh ----------------
    nx = 31
    ny = 31
    mesh = dl.UnitSquareMesh(comm, nx, ny)
    
    nodes = (nx+1)*(ny+1)
    if rank==0:
        directory_path = "../Results/Nodes_"+str(nodes)
        if not os.path.exists(directory_path):
            # Create the directory if it does not exist
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' was created.")
        else:
            print(f"Directory '{directory_path}' already exists.")
    else:
        directory_path = None
        
    comm.Barrier()
    
    
    data = generate_input_output_gempy_data(mesh=mesh,nodes=nodes, number_samples=14000, comm=comm, device=device)
    
    comm.Barrier()
    if rank==0:
        filename = directory_path + "/data_1_parameter_uniform.json"
        #filename  = "/Users/deepprakashravi/Downloads/General_python_test/Bayesian_mdoel/gempy_dino/Gempy_latest_Dino/Salt_dome/test.json"
        c ,m_data, dmdc_data = np.array(data["input"]),np.array(data["Gempy_output"]), np.array(data["Jacobian_Gempy"])
        print("Shapes-" , "Gempy Input: ", c.shape, "Gempy Output:", m_data.shape, "Jacobian shape:", dmdc_data.shape)
        with open(filename, 'w') as file:
             json.dump(data, file)
        
    

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
