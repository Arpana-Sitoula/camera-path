import copy
import glob
import shutil
from datetime import datetime

from enstools.opendata import retrieve_nwp

import json
import torch
import pygrib
import numpy as np
import netCDF4 as nc
import os

from cmcrameri import cm
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

from NeuralNetworks.TC_AR.cg_net2 import CGNet

from Data.data_util import get_grid_info_file, create_target_description_file, create_weights_file, \
    remap_icon_to_lonlat_grid, convert_to_nc


def download_tc_ar(download_path, time_step, start_time):

    download_path += "/" + str(time_step)

    # single level
    retrieve_nwp(variable=["pmsl", "tqv"],
                 model="icon",
                 grid_type="icosahedral",
                 level_type="single",
                 init_time=start_time,
                 forecast_hour=[time_step],
                 dest=download_path,
                 merge_files=False)

    # multi level
    retrieve_nwp(variable=["u", "v"],
                 model="icon",
                 grid_type="icosahedral",
                 level_type="pressure",
                 levels=850,
                 init_time=start_time,
                 forecast_hour=[time_step],
                 dest=download_path,
                 merge_files=False)

def read_grib2(filename):
    try:
        # Open the GRIB2 file and extract the data
        grbs = pygrib.open(filename)

        # Initialize an empty list to store the data
        data_list = []

        # Iterate through the GRIB messages and extract the data
        for grb in grbs:
            data = grb.values
            data_list.append(data)

        # Close the GRIB file
        grbs.close()

        # Convert the list of data arrays into a NumPy array
        numpy_array = np.array(data_list)

        return numpy_array

    except Exception as e:
        print("An error occurred:", e)
        return None

def download_and_preprocess_data(time_step=0):
    # Finds the current time, needs to be the exact hour from where the forecast was started
    current_time = datetime.now().strftime("%H")

    if int(current_time) >= 17 or int(current_time) < 5:
        start_time = 12
    else:
        start_time = 0

    # Download of the ICON data
    download_tc_ar("./Data/tc_ar_data", time_step, start_time)

    # Information needed to regrid the data from ICONs grid to a regular lon-lat-grid
    if not os.path.exists("./Data/icon_grid_0026_R03B07_G.nc"):
        grid_info_file_url = "https://opendata.dwd.de/weather/lib/cdo/icon_grid_0026_R03B07_G.nc.bz2"
        get_grid_info_file(grid_info_file_url)

    create_target_description_file(xsize=1152, ysize=768, xfirst=-180, yfirst=-90,
                                   xinc=0.3125, yinc=0.234375)
    create_weights_file()

    # The regridding
    remap_icon_to_lonlat_grid("./Data/tc_ar_data/" + str(time_step) + "/*.grib2")

def detect_features(time_step=0):
    f = open('./ZScores.json')
    zscores = json.load(f)

    indices = ["TMQ", "V850", "U850", "PSL"]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    model = CGNet()
    model.load_state_dict(torch.load("./NeuralNetworks/TC_AR/model.pth", map_location=torch.device('cpu')))

    file_name_list_raw = glob.glob("./Data/tc_ar_data/" + str(time_step) + "/*_lonlat.grib2")
    filename_list = [0, 0, 0, 0]
    for elem in file_name_list_raw:
        if "TQV" in elem:
            filename_list[0] = elem
        elif "850_V" in elem:
            filename_list[1] = elem
        elif "850_U" in elem:
            filename_list[2] = elem
        elif "PMSL" in elem:
            filename_list[3] = elem

    shutil.copyfile(filename_list[0], "./Results/TC-AR-Met3d/" + str(time_step) + ".grib2")
    convert_to_nc("./Results/TC-AR-Met3d/" + str(time_step) + ".grib2")

    data = []
    raw_data = []

    for i, filename in enumerate(filename_list):
        file = read_grib2(filename)[0]
        raw_data.append(copy.deepcopy(file))

        file -= zscores[indices[i]][0]
        file /= zscores[indices[i]][1]

        data.append(file)

    image = np.dstack(data)
    # image_shifted = np.roll(image, shift=576, axis=1)
    # image_shifted = torch.FloatTensor(np.transpose(image_shifted, (2, 0, 1)))
    image = torch.FloatTensor(np.transpose(image, (2, 0, 1)))

    raw_data = np.dstack(raw_data)
    raw_data = torch.FloatTensor(np.transpose(raw_data, (2, 0, 1)))

    output = model(image[None, :]).detach().numpy()[0]
    output = np.where(output == np.amax(output, axis=0), 1.0, 0.0)

    met3d_file = nc.Dataset("./Results/TC-AR-Met3d/" + str(time_step) + ".nc", "r+")

    met3d_file.createVariable("AR", "f4", ("time", "lat", "lon"))
    met3d_file.createVariable("TC", "f4", ("time", "lat", "lon"))
    met3d_file.createVariable("TMQ", "f4", ("time", "lat", "lon"))
    met3d_file.createVariable("U850", "f4", ("time", "lat", "lon"))
    met3d_file.createVariable("V850", "f4", ("time", "lat", "lon"))
    met3d_file.createVariable("PSL", "f4", ("time", "lat", "lon"))

    met3d_file["AR"][0] = output[2]
    met3d_file["TC"][0] = output[1]
    met3d_file["TMQ"][0] = raw_data[0]
    met3d_file["V850"][0] = raw_data[1]
    met3d_file["U850"][0] = raw_data[2]
    met3d_file["PSL"][0] = raw_data[3]

def plot_features(bg_var=0, time_step=0, wind_vectors=True):
    lons = np.linspace(-180, 180, 1152)
    lats = np.linspace(-90, 90, 768)

    indices = ["TMQ", "V850", "U850", "PSL"]
    colormaps = {0: cm.lapaz_r, 1: cm.bam_r, 2: cm.bam_r, 3: cm.turku}
    v_min_max = [[0, 100], [-40, 40], [-40, 40], [95000, 105000]][bg_var]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    data = nc.Dataset("./Results/TC-AR-Met3d/" + str(time_step) + ".nc", "r+")

    colors = ["lightsalmon", "firebrick"]

    mpl.rcParams['figure.dpi'] = 150

    scale = 4
    fig = plt.figure(time_step, figsize=(4 * scale, 3 * scale))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.pcolormesh(lons, lats, data[indices[bg_var]][0], transform=ccrs.PlateCarree(), cmap=colormaps[bg_var], vmin=v_min_max[0], vmax=v_min_max[1])

    if wind_vectors:
        X, Y = np.meshgrid(lons, lats)
        step = 5
        ax.quiver(X[::step, ::step], Y[::step, ::step], u=data[indices[2]][0][::step, ::step], v=data[indices[1]][0][::step, ::step],
                  transform=ccrs.PlateCarree(), angles="xy", units="xy", width=.15)  # , scale=0.5)

    ax.contour(lons, lats, data["TC"][0], colors=colors[0], linewidths=0.5, zorder=25)
    ax.contour(lons, lats, data["AR"][0], colors=colors[1], linewidths=0.5, zorder=20)

    ax.coastlines(resolution='110m', color="black", linewidth=0.5)
    # gl = ax.gridlines(draw_labels=False, linewidth=0.25, xlocs=[0], ylocs=[0])

    proxy_artist1 = mpatches.Rectangle((0, 0), 1, 0.1, linewidth=1, edgecolor=colors[0], facecolor='none')
    proxy_artist2 = mpatches.Rectangle((0, 0), 1, 0.1, linewidth=1, edgecolor=colors[1], facecolor='none')

    ax.legend([proxy_artist1, proxy_artist2], ["Tropical Cyclone", "Atmospheric River"], loc="lower right")

    output_dir = "./Results/TC-AR/"
    if not os.path.exists(output_dir):
        # Create a new directory because it does not exist
        os.makedirs(output_dir)
        print(f"The new directory {output_dir} is created!")

    plt.savefig(output_dir + str(time_step) + ".png", bbox_inches='tight')
    plt.clf()
    plt.close()

for i in range(0, 12, 1):
    download_and_preprocess_data(time_step=i)
    detect_features(time_step=i)
    plot_features(bg_var=0, time_step=i)