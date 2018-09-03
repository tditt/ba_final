import glob
import math
import os

import affine
import gdal
import gdalnumeric
import numpy as np
import ogr

print('GDAL_DATA' in os.environ)
# ntpath.basename("a/b/c")
tif_rel_path = '/resources/moontif/WAC_GLOBAL_P900N0000_100M.TIF'
tif_file = os.getcwd() + tif_rel_path
shape_rel_path = '\\resources\\LU78287GT_GIS\\LU78287GT_Moon2000.shp'
shape_file = os.getcwd() + shape_rel_path
gdal.UseExceptions()
dataset = gdal.Open(tif_file)
driver = ogr.GetDriverByName('ESRI Shapefile')
shape_dataset = driver.Open(shape_file, 0)
# srs = osr.SpatialReference()
# srs.ImportFromEPSG(32663)
# dataset.SetProjection(srs.ExportToWkt())
prj = dataset.GetProjection()
print('PROJ:', prj)
# if srs.IsProjected:
#     print(srs.GetAttrValue('projcs'))
# print(srs.GetAttrValue('geogcs'))
print('########################### GEOTIFF INFO ###########################\n', gdal.Info(dataset))
# datasetArray = gdalnumeric.LoadFile(tif_file)
geo_trans = dataset.GetGeoTransform()
craters = shape_dataset.GetLayer()
data_array = np.array(dataset.GetRasterBand(1).ReadAsArray())
data_array = data_array.transpose()
print("SHAPE:", data_array.shape)
forward_transform = affine.Affine.from_gdal(*geo_trans)
reverse_transform = ~forward_transform
dataset_paths = glob.glob('resources/training_north_pole/**/*.png', recursive=True)

# following variables are specific to the world file used
alt_retrieve = True
pix_per_deg = 303.23
min_lat = -90
max_lat = -60
min_lon = -180
max_lon = -150
meridian = 0
standard_parallel = 0
lon_offset = 5
lat_offset = 5


def coord2pixel_equirect(geo_coord):
    lon = geo_coord[0]
    lat = geo_coord[1]
    x = (lon - meridian) * math.cos(standard_parallel)
    y = lat - standard_parallel
    return x, y


def retrieve_pixel(geo_coord):
    if alt_retrieve:
        return retrieve_pixel_value_alt(geo_coord)
    else:
        return retrieve_pixel_value(geo_coord)


def retrieve_pixel_value(geo_coord):
    """Return floating-point value that corresponds to given point."""
    x, y = geo_coord[0], geo_coord[1]
    px, py = reverse_transform * (x, y)
    px, py = int(px + 0.5), int(py + 0.5)
    return data_array[px][py]


def retrieve_pixel_value_alt(geo_coord):
    """Return floating-point value that corresponds to given point."""
    x, y = float(geo_coord[0]), float(geo_coord[1])
    x -= min_lon
    y = max_lat - y
    px = int(math.floor(x * pix_per_deg))
    py = int(math.floor(y * pix_per_deg))
    val = data_array[px][py]
    return val


def retrieve_crater(geo_coord, radius, rotate=False):
    radius_deg, radius_km = radius[0], radius[1]
    pixels = 416
    stride_deg = 2 * radius_deg / pixels
    upper_left = [geo_coord[0] - radius_deg, geo_coord[1] + radius_deg]
    # lower_right = [geo_coord[0] - radius_deg][geo_coord[1] + radius_deg]
    x = upper_left[0]
    y = upper_left[1]

    rect = np.zeros([pixels, pixels])
    target_coordinates = np.empty([pixels, pixels], dtype=(np.dtype((float, 2))))
    rect = rect.astype(gdalnumeric.numpy.uint8)

    for x_pixel in range(0, target_coordinates.shape[0]):
        if x > 180:
            x -= 360
        elif x < -180:
            x += 360
        for y_pixel in range(0, target_coordinates.shape[1]):
            if y > 90:
                y -= 180
            elif y < -90:
                y += 180
            target_coordinates[x_pixel, y_pixel] = [x, y]
            y -= stride_deg
        y = upper_left[1]
        x += stride_deg

    if rotate:
        # get random degree from 0 to 359
        deg = (np.random.random_integers(120) - 1) * 3
        theta = np.radians(deg)
        # create rotation matrix
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        # calculate all rotated target coordinates
        for x_pixel in range(0, target_coordinates.shape[0]):
            for y_pixel in range(0, target_coordinates.shape[1]):
                [rot_x, rot_y] = geo_coord + np.matmul(R, target_coordinates[x_pixel, y_pixel] - geo_coord)
                if rot_x > 180:
                    rot_x -= 360
                elif rot_x < -180:
                    rot_x += 360
                if rot_y > 90:
                    rot_y -= 180
                elif rot_y < -90:
                    rot_y += 180
                target_coordinates[x_pixel, y_pixel] = [rot_x, rot_y]

    # fill rect with pixels of target coordinates
    for x_pixel in range(0, rect.shape[0]):
        for y_pixel in range(0, rect.shape[1]):
            rect[x_pixel, y_pixel] = retrieve_pixel(target_coordinates[x_pixel, y_pixel])

    # make up for weird alignment
    rect = np.flipud(rect)
    rect = np.rot90(rect, 3)
    return rect


def retrieve_all_craters(crater_layer):
    count = 0
    for crater in crater_layer:
        count += 1
        radius_km = crater.GetField("Radius_km")
        if radius_km > 10:
            geo_coord = [float(crater.GetField("Lon_E")), float(crater.GetField("Lat"))]
            if min_lon <= geo_coord[0] <= max_lon and min_lat <= geo_coord[1] <= max_lat:
                # if -60.0 < geo_coord[1] < 60.0: continue
                radius = [crater.GetField("Radius_deg"), radius_km]
                try:
                    rect = retrieve_crater(geo_coord, radius)
                    normalized_lon = str(geo_coord[0] + 180)
                    normalized_lat = str(geo_coord[1] + 90)
                    file_name = str(
                        str(count).zfill(5) + "_" + normalized_lat + "_" + normalized_lon + "_R" + str(
                            int(radius_km)) + ".jpeg")
                    path = os.getcwd() + '\\resources\\craters\\' + file_name
                    gdalnumeric.SaveArray(rect, path, format="JPEG")
                except:
                    print(count, geo_coord, radius, radius_km)


def retrieve_random_rectangles(size_deg=.5):
    # determines size of quadrants in degrees
    # offsets create a "padding" at the map edges which will not be used for retrieving images
    lon_area = max_lon - min_lon
    lat_area = max_lat - min_lat
    # create an array with the exact amount of elements needed
    lon_arr = np.zeros(int(math.floor(lon_area - 2 * lon_offset) / size_deg))
    # fill with valid lon-values (have to be divisible through size_deg and in between offsets)
    for i in range(0, lon_arr.shape[0]):
        lon_arr[i] = min_lon + lon_offset + i * size_deg
    # same thing for lat-values
    lat_arr = np.zeros(int(math.floor(lat_area - 2 * lat_offset) / size_deg))
    for i in range(0, lat_arr.shape[0]):
        lat_arr[i] = min_lat + lat_offset + i * size_deg
    # print("SHAPES: lon ", lon_arr.shape, " ||lat ", lat_arr.shape)
    # shuffle arrays as long as it takes
    np.random.shuffle(lon_arr)
    np.random.shuffle(lat_arr)
    while check_for_dupe_pairs(lat_arr, lon_arr):
        np.random.shuffle(lon_arr)
        np.random.shuffle(lat_arr)
    path = os.getcwd() + '\\resources\\training_north_pole\\' + str(int(math.floor(size_deg))) + '\\'
    # pick first xx items of the shuffled arrays and combine them to a coordinate
    max_pairs = lat_arr.shape[0] - 1

    for i in range(0, max_pairs):
        normalized_lon = str(lon_arr[i] + 180)
        normalized_lat = str(lat_arr[i] + 90)
        # output a square image with coordinate as the middle and size_deg*2 as side lengths
        file_name = str(
            str(i).zfill(2) + "_" + normalized_lat + "_" + normalized_lon + ".PNG")
        gdalnumeric.SaveArray(retrieve_crater([lon_arr[i], lat_arr[i]], [size_deg, 9], rotate=True), path + file_name,
                              format="PNG")


def check_for_dupe_pairs(latitudes, longitudes):
    dupes = [None] * len(dataset_paths)
    for i in range(0, len(dataset_paths)):
        filename = os.path.basename(dataset_paths[i]).split('_')
        lat = float(filename[1])
        lon = float(filename[2].rpartition('.')[0])
        dupes[i] = (lat, lon)
    for j in range(0, len(latitudes)):
        if (latitudes[j], longitudes[j]) in dupes:
            return True
    return False

