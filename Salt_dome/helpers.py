
import numpy as np

import matplotlib.pyplot as plt

import gempy as gp
import gempy_viewer as gpv


def create_initial_gempy_model(refinement, filename='prior_model.png', save=True):
    """ Create an initial gempy model objet

    Args:
        refinement (int): Refinement of grid
        save (bool, optional): Whether you want to save the image

    """
    geo_model_test = gp.create_geomodel(
    project_name='Gempy_abc_Test',  
    extent=[0, 1, -0.1, 0.1, 0, 1], 
    resolution=[100,10,100],             
    refinement=refinement,
    structural_frame= gp.data.StructuralFrame.initialize_default_structure()
    )
   
    brk1 = 0.3
    brk2 = 0.22
    brk3 = 0.2
    grad = 1.0
    gp.add_surface_points(
        geo_model=geo_model_test,
        x=[0.1, 0.5, 0.9],
        y=[0.0, 0.0, 0.0],
        z=[brk1 , brk2, brk1],
        elements_names=['surface1', 'surface1', 'surface1']
    )

    gp.add_orientations(
        geo_model=geo_model_test,
        x=[-brk3],
        y=[0.0],
        z=[brk1],
        elements_names=['surface1'],
        pole_vector=[[0, 0, grad]]
    )
    geo_model_test.update_transform(gp.data.GlobalAnisotropy.NONE)

    gp.add_orientations(
        geo_model=geo_model_test,
        x=[1 + brk3],
        y=[0.0],
        z=[brk1],
        elements_names=['surface1'],
        pole_vector=[[0, 0, grad]]
    )
    geo_model_test.update_transform(gp.data.GlobalAnisotropy.NONE)
    
    gp.compute_model(geo_model_test)
    
    picture_test = gpv.plot_2d(geo_model_test, cell_number=5, legend='force')
    if False:
        plt.show()
    if save:
        plt.savefig(filename)
    
    return geo_model_test

def create_final_gempy_model(posterior_data, refinement=7,  filename='posterior_model.png', save=True):
    """ Create an initial gempy model objet

    Args:
        refinement (int): Refinement of grid
        save (bool, optional): Whether you want to save the image

    """
    geo_model_test = gp.create_geomodel(
    project_name='Gempy_abc_Test',  
    extent=[0, 1, -0.1, 0.1, 0, 1], 
    resolution=[100,10,100],             
    refinement=refinement,
    structural_frame= gp.data.StructuralFrame.initialize_default_structure()
    )
   
    brk1 = 0.3
    brk2 = posterior_data[0]
    brk3 = 0.2
    grad = 1.0
    gp.add_surface_points(
        geo_model=geo_model_test,
        x=[00.1, 0.5, 0.9],
        y=[0.0, 0.0, 0.0],
        z=[brk1 , brk2, brk1],
        elements_names=['surface1', 'surface1', 'surface1']
    )

    gp.add_orientations(
        geo_model=geo_model_test,
        x=[-brk3],
        y=[0.0],
        z=[brk1],
        elements_names=['surface1'],
        pole_vector=[[0, 0, grad]]
    )
    geo_model_test.update_transform(gp.data.GlobalAnisotropy.NONE)

    gp.add_orientations(
        geo_model=geo_model_test,
        x=[1 + brk3],
        y=[0.0],
        z=[brk1],
        elements_names=['surface1'],
        pole_vector=[[0, 0, grad]]
    )
    geo_model_test.update_transform(gp.data.GlobalAnisotropy.NONE)

    

    gp.compute_model(geo_model_test)
    picture_test = gpv.plot_2d(geo_model_test, cell_number=5, legend='force')
    if False:
        plt.show()
    if save:
        plt.savefig(filename)
    
    return geo_model_test