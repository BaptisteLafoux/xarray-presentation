import glob
from natsort import natsorted
import os 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation
import pickle
import xarray as xr

#####
ROOT = os.path.join('example_1', 'step1_fps90_u5')
# load filenames of all piv time step 
piv_frames = natsorted(glob.glob(os.path.join(ROOT, 'frames', '*.npy')))

# load metadata file in a dictionary 
with open(os.path.join(ROOT, 'step1_fps90_u5_metadata'), 'rb') as file: 
    metadata = pickle.load(file)

#####
# load x-y coordinates 
x, y = np.load(os.path.join(ROOT, 'coords.npy'))

# load velocity vectors from piv results in an array
U = np.array([np.load(piv_frame) for piv_frame in piv_frames])
print(f'Shape of velocity array : {U.shape} > (time, direction, y, x)')

ux, uy = U[:, 0], U[:, 1]

#####
# compute some values 
curl = np.gradient(uy, axis=2) - np.gradient(ux, axis=1)
div  = np.gradient(ux, axis=2) + np.gradient(uy, axis=1)
vel  = np.sqrt(ux ** 2 + uy ** 2)

#####
# create a dictonary with the data you want to save 
data_dict = dict(
    
    ux = (['time', 'y', 'x'], ux), # >>> even better : (['time', 'y', 'x'], ux.astype(float32))
    uy = (['time', 'y', 'x'], uy),

    vel  = (['time', 'y', 'x'], vel),

    curl = (['time', 'y', 'x'], curl),
    div  = (['time', 'y', 'x'], div),

    )

#####
# create a coordinate dictonary, containing the coordinates of your data 
coords_dict = dict(

x = ('x', np.unique(x)),
y = ('y', np.unique(y)),

time = ('time', np.arange(metadata['n_frames']) * metadata['dt']),

height  = ('height',  [metadata['step']]),
voltage = ('voltage', [metadata['voltage']])

)

####
# create your Xarray dataset 
ds = xr.Dataset(data_vars=data_dict, coords=coords_dict, attrs=metadata)
# encoding for compression (optional)
encoding = {var: dict(zlib=True, complevel=5) for var in ds.data_vars}
# saving as a netcdf file
ds.to_netcdf(os.path.join('example_1', 'example_ds.nc'), encoding=encoding)
######

ds = xr.open_dataset(os.path.join('example_1', 'example_ds.nc'))

time_idx = 200

ds_ = ds.isel(x=slice(35, 155), y=slice(8, 88)) #to keep only a region of interest in the images
ds_ = ds_.isel(time=time_idx)

fig, ax = plt.subplots(figsize=(7, 3.5)) 

variable = 'vel'
pcm = ax.pcolormesh(ds_.x, ds_.y, abs(ds_[variable]))

ax.axis('scaled');

ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
cbar = fig.colorbar(pcm)
cbar.set_label(variable, rotation=90)
fig.tight_layout()


fig, ax = plt.subplots()

ds_ = ds.isel(x=slice(35, 155), y=slice(8, 88))
pcm = ax.pcolormesh(ds_.x, ds_.y, ds_[variable].isel(time=0), vmin=0, cmap='Reds')
ax.axis('scaled');

ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
cbar = fig.colorbar(pcm)
cbar.set_label(variable, rotation=90)
fig.tight_layout()

def animate(i):
   pcm.set_array(ds_[variable].isel(time=i).to_numpy().flatten())

anim = animation.FuncAnimation(fig, animate, interval= 1000 / (5 * ds.fps), frames=ds.n_frames)
#anim.save('test.gif')
