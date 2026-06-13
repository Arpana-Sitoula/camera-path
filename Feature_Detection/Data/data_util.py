import os
import bz2
import glob
import numpy as np
import netCDF4 as nc
import urllib.request

from metpy.units import units
from metpy.calc import dewpoint_from_relative_humidity, specific_humidity_from_dewpoint

from cdo import *

cdo = Cdo()


def create_folder(download_path):
    if not os.path.exists(download_path):
        # Create a new directory because it does not exist
        os.makedirs(download_path)
        print(f"The new directory {download_path} is created!")


def convert_to_nc(path_grib):
    print()
    for file_grib in glob.glob(path_grib):
        print(file_grib)
        filename, _ = os.path.splitext(file_grib)
        file_nc = filename + ".nc"
        cdo.copy(input=file_grib, output=file_nc, options="-f nc")
        print("successfully converted to", file_nc)


def get_grid_info_file(url):
    file_name = "./Data/" + os.path.split(url)[1]

    # Set the download path
    download_path = os.path.join(os.getcwd(), file_name)

    # Download the file
    try:
        urllib.request.urlretrieve(url, download_path)
        print(f"Downloaded {file_name} successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Decompress the downloaded file
    output_file = file_name.replace(".bz2", "")
    with bz2.BZ2File(file_name, 'rb') as source, open(output_file, 'wb') as dest:
        dest.write(source.read())

    print(f"Decompressed {file_name} to {output_file}.")


def create_weights_file():
    cdo.gennn("./Data/target_grid_description.txt", input="./Data/icon_grid_0026_R03B07_G.nc", output="./Data/weights.nc")


def create_target_description_file(gridtype='lonlat', xsize=1152, ysize=768, xfirst=-180, yfirst=-90,
                                   xinc=0.3125, yinc=0.234375):
    f = open("./Data/target_grid_description.txt", "w")
    f.write(f'''# CDO grid description file
            gridtype = {gridtype}
            xsize = {xsize}
            ysize = {ysize}
            xfirst = {xfirst}
            xinc = {xinc}
            yfirst = {yfirst}
            yinc = {yinc}''')
    f.close()


def remap_icon_to_lonlat_grid(icon_grid_files):
    for input_path in glob.glob(icon_grid_files):

        if "lonlat" == input_path.split("_")[-1].split(".")[0]:
            return

        print(f"remap {input_path}:")
        path, input_file = os.path.split(input_path)
        file, extension = os.path.splitext(input_file)
        output_dir = path + "/"
        output_path = output_dir + file + "_lonlat" + extension

        if not os.path.exists(output_dir):
            # Create a new directory because it does not exist
            os.makedirs(output_dir)
            print(f"The new directory {output_dir} is created!")

        cdo.remap("./Data/target_grid_description.txt,./Data/weights.nc", input=input_path, output=output_path, options="-f grb2")


def calc_specific_humidity(filepath):
    for input_path in glob.glob(filepath):
        ds = nc.Dataset(input_path, mode='a')
        ds["plev"][:] /= 100

        pressure = np.stack([np.full((720, 1440), 1000), np.full((720, 1440), 950),
                             np.full((720, 1440), 925), np.full((720, 1440), 850),
                             np.full((720, 1440), 800), np.full((720, 1440), 700),
                             np.full((720, 1440), 600)])[None, :]

        pressure = units.hPa * pressure

        temperature = units.degK * ds["t"][:]
        rel_humidity = units.percent * ds["r"][:]

        dewpoint = dewpoint_from_relative_humidity(temperature, rel_humidity)
        specific_humidity = np.array(specific_humidity_from_dewpoint(pressure, dewpoint).to('kg/kg'))

        ds.createVariable("q", np.float64, ('time', "plev", 'lat', 'lon'))
        ds["q"][:] = specific_humidity[:]

        ds.renameDimension("plev", "level")
        ds.renameVariable("plev", "level")

        ds.renameDimension("lat", "latitude")
        ds.renameVariable("lat", "latitude")

        ds.renameDimension("lon", "longitude")
        ds.renameVariable("lon", "longitude")
