import sdf_helper
import os
import shutil
import numpy as np
import xarray as xr
from typing import List,Dict
import time
def copy_to_dev_shm(source_path):
    """
    Copies a file to /dev/shm (RAM-based temporary storage).
    """
    try:
        # Validate source file
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Ensure /dev/shm exists and is writable
        shm_path = "/dev/shm"
        if not os.path.isdir(shm_path):
            raise NotADirectoryError(f"{shm_path} does not exist on this system.")
        if not os.access(shm_path, os.W_OK):
            raise PermissionError(f"No write permission to {shm_path}")

        # Build destination path
        dest_path = os.path.join(shm_path, f'{int(time.time())}_{os.path.basename(source_path)}')

        # Copy file (preserves permissions but not timestamps)
        shutil.copy(source_path, dest_path)

        print(f"File copied to: {dest_path}")
        return dest_path

    except Exception as e:
        print(f"Error: {e}")
        return None

def cleanup_file(path):
    """Delete the file if it exists."""
    try:
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted '{path}' from /dev/shm")
    except Exception as e:
        print(f"Warning: Could not delete '{path}': {e}")



def read_dat(dat_name='',shape=()):
    print('Read %s' %(dat_name))
    print(shape)
    data=np.fromfile(dat_name,dtype=np.float64)
    data=data.reshape(np.flip(shape))
    data=data.transpose()
    return data




def write_fields_to_nc(
    field_dict_list: List[Dict],
    coordinate_dict_list: List[Dict],
    nc_name="fields.nc",working_dir=''
    ):
    """
    Write fields and coordinates to a NetCDF file.
    Parameters
    ----------
    field_dict_list : List[Dict]
        List of dictionaries containing field data and metadata. Each dictionary should have the following keys:
        field_dict_list=[
            {
                'name': '',  #str, name of the field component
                'data': field,  #np.ndarray
                'units': '',
                'long_name': ''
            },
            ...
            }
        ]
    coordinate_dict_list : List[Dict]
        List of dictionaries containing coordinate data and metadata. Each dictionary should have the following keys:
        coordinate_dict_list=[
            {
                'name': '',  #str, name of the coordinate axis
                'coordinate': coordinate_array,  #np.ndarray
                'units': '',
                'long_name': ''
            },
            ...
            ]
        ]
    """
    ndim=len(coordinate_dict_list)
    shape=[]
    coordinate_name_list=[]
    coords={}
    for dim in range(ndim):
        coordinate_dict=coordinate_dict_list[dim]
        assert 'name' in coordinate_dict, f"Coordinate dictionary at index {dim} must have a 'name' key."
        assert 'coordinate' in coordinate_dict, f"Coordinate dictionary at index {dim} must have a 'coordinate' key."
        coordinate_dict['coordinate']=np.asarray(coordinate_dict['coordinate'],dtype=np.float64).flatten()
        shape.append(coordinate_dict['coordinate'].size)
        coordinate_name_list.append(coordinate_dict['name'])
        coords[coordinate_dict['name']]=(
            [coordinate_dict['name']],
            coordinate_dict['coordinate'],
            {'units': coordinate_dict.get('units', ''), 'long_name': coordinate_dict.get('long_name', '')}
        )
    shape=tuple(shape)
    data_vars={}
    encoding={}
    for field_dict in field_dict_list:
        assert 'name' in field_dict, "Field dictionary must have a 'name' key."
        assert 'data' in field_dict, f"Field dictionary for {field_dict['name']} must have a 'data' key."
        field_dict['data']=np.asarray(field_dict['data'])
        assert field_dict['data'].shape == shape, f"Field data shape {field_dict['data'].shape} does not match coordinate shape {shape} for field {field_dict['name']}."
        data_vars[field_dict['name']]=(
            coordinate_name_list,
            field_dict['data'],
            {'units': field_dict.get('units', ''), 'long_name': field_dict.get('long_name', '')}
        )
        encoding[field_dict['name']]={'zlib': True, 'complevel': 5}
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords
    )
    nc_path=os.path.join(working_dir, nc_name+'.nc')
    ds.to_netcdf(path=nc_path, mode='w', format="NETCDF4", engine='h5netcdf',encoding=encoding)
    print(nc_path)
    return nc_path

def read_nc(nc_name='',key_name_list=[]):
    print('Read %s' %(nc_name))
    nc=xr.open_dataset(filename_or_obj=nc_name)
    print(f'Available keys: {list(nc.keys())}')
    data_dict={}
    print(nc.coords)
    for coord_name in nc.coords.keys():
        print(f'{coord_name}: {nc.coords[coord_name]}')
        data_dict[coord_name]=nc.coords[coord_name].to_numpy()
    print(nc.data_vars)
    for key_name in key_name_list:
        print(f'{key_name}: {nc[key_name]}')
        data_dict[key_name]=nc.data_vars[key_name].to_numpy()
    return data_dict

def read_sdf(sdf_name='',block_name_list=[],Grid_name='Grid_mid',use_shm=True):
    #https://github.com/epochpic/sdf-xarray
    if use_shm:
        fname=copy_to_dev_shm(sdf_name)
    else:
        fname=sdf_name
    
    data_dict=read_nc(nc_name=fname,key_name_list=block_name_list)
    
    try:
        data_dict['x']=data_dict['X_'+Grid_name]
        print(f'x,shape={data_dict["x"].shape}')
    except:
        pass
    try:
        data_dict['y']=data_dict['Y_'+Grid_name]
        print(f'y,shape={data_dict["y"].shape}')
    except:
        pass
    try:
        data_dict['z']=data_dict['Z_'+Grid_name]
        print(f'z,shape={data_dict["z"].shape}')
    except:
        pass
    print(f'Available keys: {list(data_dict.keys())}')
    return data_dict
    sdf=sdf_helper.getdata(fname=fname)
    
    data_dict={}
    data_dict['time']=sdf.__dict__['Header']['time']
    try:
        data_dict['x']=sdf.__dict__[Grid_name].data[0]
        print(f'x,shape={data_dict["x"].shape}')
    except:
        pass
    try:
        data_dict['y']=sdf.__dict__[Grid_name].data[1]
        print(f'y,shape={data_dict["y"].shape}')
    except:
        pass
    try:
        data_dict['z']=sdf.__dict__[Grid_name].data[2]
        print(f'z,shape={data_dict["z"].shape}')
    except:
        pass
    for block_name in block_name_list:
        data_dict[block_name]=sdf.__dict__[block_name].data
        print(f'{block_name},shape={data_dict[block_name].shape}')
    if use_shm:
        cleanup_file(fname)
    return data_dict

if __name__ == "__main__":
    #read_nc(nc_name='/scratch/gpfs/MIKHAILOVA/zl8336/Small_a0/test/collision/EE_collision/cpl=5000,ppl=2000/fields0001.sdf',key_name_list=[])
    read_sdf(sdf_name='/scratch/gpfs/MIKHAILOVA/zl8336/Small_a0/test/collision/all_collision/cpl=5000,ppl=2000/fields00001.sdf',block_name_list=['Electric_Field_Ey'],use_shm=True)


