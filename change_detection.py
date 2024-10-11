
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.features import geometry_mask
from rasterio.sample import sample_gen
from shapely.geometry import LineString

# Lade die Geländemodelle und das Shapefile
# Shisper
#dem1_path = "/home/mela/samples/wsl_ames_test_shi_2017/PSScene_overlap_10_converg_4.0_map/final_rpc_stereo/mosaic_dems/point_cloud_mosaic-DEM.tif"
#dem2_path = "/home/mela/samples/wsl_ames_test_shi_2019/PSScene_overlap_10_converg_4.0_map/final_rpc_stereo/mosaic_dems/point_cloud_mosaic-DEM.tif"
dem1_path = "/home/mela/samples/change_detection_shisper/data/2017_point_cloud_mosaic-DEM_output_COP30.tif-adj.tif_utm_nuth_x-3.28_y+2.48_z+3.25_align.tif"
dem2_path = "/home/mela/samples/change_detection_shisper/data/2019_point_cloud_mosaic-DEM_output_COP30.tif-adj.tif_utm_nuth_x-2.27_y+2.85_z+0.29_align.tif"
mask_shapefile_path = "/home/mela/samples/change_detection_shisper/glacier_mask/mask_shisper_4_dod.shp"
hillshade_dem_path = "/home/mela/samples/wsl_ames_test_shi_2017/PSScene_overlap_10_converg_4.0_map/refdem/output_COP30.tif-adj.tif_utm.tif"  # Pfad zum neuen DEM
profile_shapefile_path = "/home/mela/samples/change_detection_shisper/height_profile/shisper_profile.shp"  # Pfad zur Polyline-Shapefile
output_path_print = "/home/mela/samples/change_detection_shisper/out"
year1 = '2017'
year2 = '2019'
k = 50

# # Lade die Geländemodelle und das Shapefile
# dem1_path = "/home/mela/samples/wsl_ames_test_bov_2019/201908_PSS4_overlap_10_converg_4.0_map/final_rpc_stereo/mosaic_dems/point_cloud_mosaic-DEM.tif"
# dem2_path = "/home/mela/samples/wsl_ames_test_bov_2020/202009_PSS4_overlap_10_converg_4.0_map/final_rpc_stereo/mosaic_dems/point_cloud_mosaic-DEM.tif"
# hillshade_dem_path = "/home/mela/samples/wsl_ames_test_bov_2021/202109_PSS4_overlap_10_converg_4.0_map/refdem/output_COP30.tif-adj.tif_utm.tif"  # Pfad zum neuen DEM
# mask_shapefile_path = "/home/mela/samples/change_detection_bov/glaciers_jotunheimen_mask/jotunheimen_west_rect.shp"  # Pfad zur Polyline-Shapefile
# output_path_print = "/home/mela/samples/change_detection_bov/out"
# profile_shapefile_path = "/home/mela/samples/change_detection_shisper/height_profile/shisper_profile.shp"  # Pfad zur Polyline-Shapefile
# dem_bov_2019 = "/home/mela/samples/wsl_ames_test_bov_2019/201908_PSS4_overlap_10_converg_4.0_map/final_rpc_stereo/mosaic_dems/point_cloud_mosaic-DEM_output_COP30.tif-adj.tif_utm_nuth_x-9.71_y+8.02_z-26.87_align.tif"
# dem_bov_2020 = "/home/mela/samples/wsl_ames_test_bov_2020/202009_PSS4_overlap_10_converg_4.0_map/final_rpc_stereo/mosaic_dems/point_cloud_mosaic-DEM_output_COP30.tif-adj.tif_utm_nuth_x+1.73_y+2.00_z-0.14_align.tif"
# dem_bov_2021 = "/home/mela/samples/wsl_ames_test_bov_2021/202109_PSS4_overlap_10_converg_4.0_map/final_rpc_stereo/mosaic_dems/point_cloud_mosaic-DEM_output_COP30.tif-adj.tif_utm_nuth_x-9.13_y-3.32_z-25.04_align.tif"
# year1 = '2019'
# year2 = '2021'
# k = 5
# dem1_path = dem_bov_2019
# dem2_path = dem_bov_2021



# IQR 1.5 Filter
def removeOutliers(arr, outlierConstant = 1.5):
    lower_quartile  = np.nanpercentile(arr, 25)
    upper_quartile  = np.nanpercentile(arr, 75)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    lower_bound = lower_quartile - IQR
    upper_bound = upper_quartile + IQR
    filtered_data = arr.copy()

    filter_mask = (arr < lower_bound) | (arr > upper_bound)
    filtered_data[filter_mask] = np.nan

    # Number of original valid values (without NaN)
    num_original_valid_values = np.sum(~np.isnan(arr))

    # Number of newly filtered values (excluding the original NaN)
    num_filtered_values = np.sum(filter_mask & ~np.isnan(arr))

    percent_filtered = (num_filtered_values / num_original_valid_values) * 100
    print("Remove coarse outliers from DoD using IQR filtering with constant (k):", outlierConstant)
    return filtered_data, percent_filtered





# Funktion zum Laden eines Rasters inklusive NoData-Wert
def load_raster(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Lade nur das erste Band (Höhenwerte)
        nodata = src.nodata  # NoData-Wert abfragen
        transform = src.transform
        crs = src.crs
        profile = src.profile
    return data, transform, crs, profile, nodata

# Funktion zum Resampling eines Rasters auf ein Zielraster
def resample_raster(source, source_transform, source_crs, target_shape, target_transform, target_crs, nodata_value):
    destination = np.empty(target_shape, dtype=source.dtype)
    reproject(
        source,
        destination,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear
    )
    destination[destination == nodata_value] = np.nan
    return destination

# Funktion zum Anwenden der Maske (Shapefile) auf das Raster
def apply_mask(raster_data, transform, shapefile, nodata_value):
    shapes = [geom for geom in shapefile.geometry]
    out_image, out_transform = mask(raster_data, shapes, crop=True, nodata=nodata_value)
    out_image[out_image == nodata_value] = np.nan
    return out_image[0], out_transform






# Geländemodelle und Shapefile laden
dem1, dem1_transform, dem1_crs, dem1_profile, dem1_nodata = load_raster(dem1_path)
dem2, dem2_transform, dem2_crs, dem2_profile, dem2_nodata = load_raster(dem2_path)
mask_shapefile = gpd.read_file(mask_shapefile_path)

# NoData-Werte durch NaN ersetzen
if dem1_nodata is not None:
    dem1[dem1 == dem1_nodata] = np.nan
if dem2_nodata is not None:
    dem2[dem2 == dem2_nodata] = np.nan

# Maskierung auf dem Raster anwenden
with rasterio.open(dem1_path) as src1:
    masked_dem1, masked_dem1_transform = apply_mask(src1, dem1_transform, mask_shapefile, dem1_nodata)

# Resample DEM2 auf die Auflösung von DEM1
dem2_resampled = resample_raster(dem2, dem2_transform, dem2_crs, masked_dem1.shape, masked_dem1_transform, dem1_crs, dem2_nodata)

# Berechne die Differenz
dem_diff = dem2_resampled - masked_dem1
dem_diff, percent = removeOutliers(dem_diff, k)

# Lade das DEM für die Hillshade-Berechnung
hillshade_dem, hillshade_transform, hillshade_crs, hillshade_profile, hillshade_nodata = load_raster(hillshade_dem_path)

# Schummerung (Hillshade) für das unabhängige DEM erstellen
def hillshade(array, azimuth=315, angle_altitude=45):
    x, y = np.gradient(array)
    slope = np.pi / 2. - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)
    azimuth_rad = azimuth * np.pi / 180.
    altitude_rad = angle_altitude * np.pi / 180.
    shaded = np.sin(altitude_rad) * np.sin(slope) + np.cos(altitude_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect)
    return 255 * (shaded + 1) / 2

# Hillshade für das unabhängige DEM berechnen
hillshade_array = hillshade(hillshade_dem)

# Visualisierung: Differenz und Hillshade
plt.figure(figsize=(10, 10))

# Bereiche außerhalb von dem_diff mit NaN versehen
dem_diff_masked = np.where(np.isnan(dem_diff), np.nan, dem_diff)



#########################################################

# Hintergrund-Hillshade über das gesamte Fenster
plt.imshow(hillshade_array, cmap='gray', alpha=0.5, extent=(
    hillshade_transform[2],
    hillshade_transform[2] + hillshade_transform[0] * hillshade_dem.shape[1],
    hillshade_transform[5] + hillshade_transform[4] * hillshade_dem.shape[0],
    hillshade_transform[5]
))  # Alpha-Wert für Transparenz


# Differenz darstellen
diff_img = plt.imshow(dem_diff_masked, cmap='RdYlBu_r', alpha=0.8, extent=(
    masked_dem1_transform[2],
    masked_dem1_transform[2] + masked_dem1_transform[0] * masked_dem1.shape[1],
    masked_dem1_transform[5] + masked_dem1_transform[4] * masked_dem1.shape[0],
    masked_dem1_transform[5]
))

# Farbbalken und Titel hinzufügen
cbar = plt.colorbar(diff_img, label='Elevation Change (m)', orientation='vertical', shrink=0.8)
#cbar.ax.set_ylim(-100, 100)  # Setze die Grenzen des Colorbars
plt.clim(-150,150)
plt.title('DoD ' + year1 + '/' + year2)



if output_path_print is not None:
    plt.savefig(output_path_print + "_dod_" + year1 + "_" + year2 + ".png", dpi=300, bbox_inches='tight')
plt.close()

##########################################

# Histogramm der Veränderungen
plt.figure(figsize=(5, 5))
valid_diff = dem_diff_masked[~np.isnan(dem_diff_masked)]  # Filtere NaN-Werte
plt.hist(valid_diff, bins=50, color='blue', alpha=0.8, edgecolor = 'w')

# Berechne Statistik
mean_diff = np.nanmean(valid_diff)
median_diff = np.nanmedian(valid_diff)
min_diff = np.nanmin(valid_diff)
max_diff = np.nanmax(valid_diff)

# Annotiere Statistik im Histogramm
plt.axvline(mean_diff, color='red', linestyle='dashed', linewidth=1, label=f'mean: {mean_diff:.2f}')
plt.axvline(median_diff, color='orange', linestyle='dashed', linewidth=1, label=f'median: {median_diff:.2f}')
plt.axvline(min_diff, color='green', linestyle='dashed', linewidth=1, label=f'min: {min_diff:.2f}')
plt.axvline(max_diff, color='purple', linestyle='dashed', linewidth=1, label=f'max: {max_diff:.2f}')

# Legende und Titel hinzufügen
plt.legend()
plt.title('Histogramm of Elevation Changes')
plt.xlabel('Elevation Change (m)')
plt.ylabel('Number of Pixels')
plt.grid()
#plt.show()

if output_path_print is not None:
    plt.savefig(output_path_print + "_histogram_elev_changes_" + year1 + "_" + year2 + ".png", dpi=300, bbox_inches='tight')
plt.close()




###################################


# Funktion zum Laden eines Rasters
def load_raster(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Nur das erste Band laden (Höhenwerte)
        transform = src.transform
        crs = src.crs
        profile = src.profile
        nodata = src.nodata
    return data, transform, crs, profile, nodata

# Funktion zum Extrahieren der Höhen entlang einer Polyline
def extract_profile_from_dem(dem_path, polyline, num_points=100):
    with rasterio.open(dem_path) as dem:
        # Generiere Punkte entlang der Polyline
        line = polyline.geometry.iloc[0]  # Die erste Zeile in Geometrie nehmen
        points = [line.interpolate(i / num_points, normalized=True) for i in range(num_points + 1)]
        
        # Extrahiere Höhenwerte entlang der Polyline
        coords = [(point.x, point.y) for point in points]
        heights = [val[0] for val in dem.sample(coords)]  # Werte aus DEM extrahieren
        
        # Berechne die Distanzen entlang der Polyline für x-Achse
        distances = [line.project(point) for point in points]  # Entfernungen entlang der Linie
        
    return distances, heights

# Shapefile laden (Polyline für Höhenprofil)
polyline = gpd.read_file(profile_shapefile_path)


# Höhenprofile für beide DEMs extrahieren
distances_dem1, heights_dem1 = extract_profile_from_dem(dem1_path, polyline)
distances_dem2, heights_dem2 = extract_profile_from_dem(dem2_path, polyline)

# Plotten des Höhenprofils
plt.figure(figsize=(5, 5))
plt.plot(distances_dem1, heights_dem1, label=year1, color='blue')
plt.plot(distances_dem2, heights_dem2, label=year2, color='red')
plt.fill_between(distances_dem1, heights_dem1, heights_dem2, color='gray', alpha=0.3, label='diff')
plt.xlabel('Elevation Profile (m)')
plt.ylabel('Elevation (m)')
plt.title('Elevation Profile Deviation ' + year1 + '/' + year2)
plt.legend()
plt.grid(True)
#plt.show()

if output_path_print is not None:
    plt.savefig(output_path_print + "_elevation_profile_" + year1 + "_" + year2 + ".png", dpi=300, bbox_inches='tight')
plt.close()
