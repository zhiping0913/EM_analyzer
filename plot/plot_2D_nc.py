import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336')
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/EM_analyzer')
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/EM_analyzer/plot')
import os
import numpy as np
import scipy.constants as C
from EM_analyzer.read_write import read_nc
from plot.plot_2D import plot_2D_field

working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/Curved_surface/a0=20/2D/K=-0.002,D=0.02,L=0.00'
theta_degree=45
theta_rad=np.radians(theta_degree)

laser_lambda = 0.8*C.micron		# Laser wavelength
laser_f0=1/laser_lambda
laser_k0=2*C.pi*laser_f0
laser_omega0=(2*C.pi*C.speed_of_light)/(laser_lambda)
laser_T0=laser_lambda/C.speed_of_light
laser_Bc=(C.m_e*laser_omega0)/(C.elementary_charge)
laser_Ec=laser_Bc*C.speed_of_light   #4.013376e+12V/m
laser_Sc=C.epsilon_0*C.speed_of_light*laser_Ec**2/2   #Sc=ε0·c·Ec^2/2=Ec·Bc/(2*μ0)   1.327e+18 W/m^2
laser_Nc=laser_omega0**2*C.m_e*C.epsilon_0/C.elementary_charge**2


target_N=200


laser_lambda_M=laser_lambda/np.cos(theta_rad)
laser_T0_M=laser_lambda_M/C.speed_of_light
laser_f0_M=laser_f0*np.cos(theta_rad)
laser_k0_M=laser_k0*np.cos(theta_rad)
laser_Bc_M=laser_Bc*np.cos(theta_rad)
laser_Ec_M=laser_Ec*np.cos(theta_rad)
laser_Sc_M=laser_Sc*np.cos(theta_rad)**2


def plot_2D_nc(nc_name:str,working_dir:str):
    key_name_list=[Block['name'] for Block in Block_list]
    data_dict=read_nc(nc_name=os.path.join(working_dir,nc_name),key_name_list=key_name_list)
    coord1=data_dict[coordinate_name_list[0]['name']]/coordinate_name_list[0]['normalize']
    coord2=data_dict[coordinate_name_list[1]['name']]/coordinate_name_list[1]['normalize']
    for Block in Block_list:
        name=Block['name']
        field=data_dict[name]/Block['normalize']
        plot_2D_field(
            field=field,x_coordinate=coord1,y_coordinate=coord2,
            name=f'{nc_name}_{name}',label=Block['label'],
            xlabel=coordinate_name_list[0]['label'],ylabel=coordinate_name_list[1]['label'],
            return_fig=False,working_dir=working_dir,
            cmap=Block.get('cmap', 'RdBu'),profile_at_x=[0],profile_at_y=[0],
            step_x=10,step_y=10,
            #xmin=5, xmax=10, ymin=-0.5, ymax=0.5,
            vmin=-30,vmax=30,
            )
Block_list=[
    {'name':'Ey','cmap':'RdBu','normalize':laser_Ec,'label':'a=Ey/Ec'},
    {'name':'Ex','cmap':'RdBu','normalize':laser_Ec,'label':'a=Ex/Ec'},
    #{'name':'Ez','cmap':'RdBu','normalize':laser_Ec,'label':'a=Ez/Ec'},
    #{'name':'Bx','cmap':'RdBu','normalize':laser_Bc,'label':'a=Bx/Bc'},
    #{'name':'By','cmap':'RdBu','normalize':laser_Bc,'label':'a=By/Bc'},
    {'name':'Bz','cmap':'RdBu','normalize':laser_Bc,'label':'a=Bz/Bc'},
    #{'name':'N_e','cmap':'RdBu','normalize':laser_Nc,'label':'Ne/Nc','cmap':'Reds'},
    #{'name':'Jx_e','cmap':'RdBu','normalize':-laser_Nc*C.elementary_charge*C.speed_of_light,'label':'Jx_e/(-Nc * e * c)'},
    #{'name':'Jy_e','cmap':'RdBu','normalize':-laser_Nc*C.elementary_charge*C.speed_of_light,'label':'Jy_e/(-Nc * e * c)'},
    #{'name':'Jz_e','cmap':'RdBu','normalize':-laser_Nc*C.elementary_charge*C.speed_of_light,'label':'Jz_e/(-Nc * e * c)'},
    #{'name':'Sx','cmap':'RdBu','normalize':laser_Sc,'label':'Sx/Sc'},
    #{'name':'Sy','cmap':'RdBu','normalize':laser_Sc,'label':'Sy/Sc'},
    #{'name':'Sz','cmap':'RdBu','normalize':laser_Sc,'label':'Sz/Sc'},

] 
coordinate_name_list=[
    #{'name':'time','normalize':laser_T0_M,'label':'t_M/T0'},
    {'name':'x','normalize':laser_lambda,'label':'x/λ0'},
    {'name':'y','normalize':laser_lambda,'label':'y/λ0'},
]

plot_2D_nc(
    nc_name='fields0001_250cpl_transmission.nc',
    working_dir=working_dir
)



