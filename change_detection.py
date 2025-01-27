
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from rasterio.features import geometry_mask
from rasterio.sample import sample_gen
from shapely.geometry import LineString
from mpl_toolkits import axes_grid1

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
    print("Remove coarse outliers from DoD using IQR filtering with constant (k):", outlierConstant, "percentage filtered:",  percent_filtered)
    return filtered_data, percent_filtered


# Schummerung (Hillshade) für das unabhängige DEM erstellen
def hillshade(array, azimuth=315, angle_altitude=45):
    x, y = np.gradient(array)
    slope = np.pi / 2. - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)
    azimuth_rad = azimuth * np.pi / 180.
    altitude_rad = angle_altitude * np.pi / 180.
    shaded = np.sin(altitude_rad) * np.sin(slope) + np.cos(altitude_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect)
    return 255 * (shaded + 1) / 2

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

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)



# Lade die Geländemodelle und das Shapefile
# Shisper
# dem1_path = "/home/mela/samples/wsl_ames_test_shi_2017_ellips/PSScene_overlap_10_converg_4.0_map/final_rpc_stereo/mosaic_dems/point_cloud_mosaic-DEM_clipped.tif_utm_output_COP30.tif-adj_clipped.tif_utm_nuth_x-2.03_y+3.71_z+2.81_align.tif"
# dem2_path = "/home/mela/samples/wsl_ames_test_shi_2019_ellips/PSScene_overlap_10_converg_4.0_map/final_rpc_stereo/mosaic_dems/point_cloud_mosaic-DEM_clipped.tif_utm_output_COP30.tif-adj_clipped.tif_utm_nuth_x-1.83_y+6.16_z+0.33_align.tif"
# mask_shapefile_path = "/home/mela/samples/change_detection_shisper/glacier_mask/mask_shisper_4_dod.shp"
# hillshade_dem_path = "/home/mela/samples/wsl_ames_test_shi_2019_ellips/PSScene_overlap_10_converg_4.0_map/refdem/output_COP30.tif-adj_clipped.tif_utm.tif"  # Pfad zum neuen DEM
# profile_shapefile_path = "/home/mela/samples/change_detection_shisper/height_profile/shisper_profile.shp"  # Pfad zur Polyline-Shapefile
# output_path_print = "/home/mela/samples/change_detection_shisper/out_ellips"
# year1 = '2017'
# year2 = '2019'
# k = 50
# profile = True
# glacier_outline = False




# # Lade die Geländemodelle und das Shapefile
dem_2021 = "/home/mela/samples/wsl_ames_test_bov_2021_ellips_2/202109_PSS4_overlap_10_converg_4.0_map/final_rpc_stereo/mosaic_dems/point_cloud_mosaic-DEM_clipped.tif_utm_output_COP30.tif-adj_clipped.tif_utm_nuth_x-1.54_y+1.52_z+1.49_align.tif"
dem_alos30m = "/home/mela/samples/change_detection_bov/alos_dem30m/output_AW3D30E_utm.tif"
hillshade_dem_path = "/home/mela/samples/change_detection_bov/alos_dem30m/output_AW3D30E_utm.tif"
mask_shapefile_path = "/home/mela/samples/change_detection_bov/glaciers_jotunheimen_mask/outline_roi_ndh_planet.shp"  # Pfad zur Polyline-Shapefile
output_path_print = "/home/mela/samples/change_detection_bov/out_ellips"
year1 = 'AW3D30'
year2 = '2021'
k = 5
profile = False
profile_shapefile_path = ""
dem1_path = dem_alos30m
dem2_path = dem_2021 # dem_2021
glacier_outline = True
glacier_outline_shape = "/home/mela/samples/change_detection_bov/glaciers_jotunheimen_mask/glacier_mask_jotunheimen.shp"



# dem2- dem1



# ensure common extent
from rasterio import features
from rasterio.mask import mask
# the first one is your raster on the right
# and the second one your red raster
with rasterio.open(dem2_path) as src, \
        rasterio.open(dem1_path) as src_to_crop:
    src_affine = src.meta.get("transform")

    # Read the first band of the "mask" raster
    band = src.read(1)
    # Use the same value on each pixel with data
    # in order to speedup the vectorization
    band[np.where(band!=src.nodata)] = 1

    geoms = []
    for geometry, raster_value in features.shapes(band, transform=src_affine):
        # get the shape of the part of the raster
        # not containing "nodata"
        if raster_value == 1:
            geoms.append(geometry)

    # crop the second raster using the
    # previously computed shapes
    out_img, out_transform = mask(
        dataset=src_to_crop,
        shapes=geoms,
        crop=True,
    )

    # save the result
    # (don't forget to set the appropriate metadata)
    with rasterio.open(
        dem1_path + "_clipped_dem2.tif",
        'w',
        driver='GTiff',
        height=out_img.shape[1],
        width=out_img.shape[2],
        count=src.count,
        crs=src.crs,
        dtype=out_img.dtype,
        transform=out_transform,
    ) as dst:
        dst.write(out_img)
dem1_path = dem1_path + "_clipped_dem2.tif"








    




if profile:
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


########################
# Berechne die Differenz
########################
dem_diff = dem2_resampled - masked_dem1
dem_diff, percent = removeOutliers(dem_diff, k) # je größer K, desto weniger wird gefiltert

# Lade das DEM für die Hillshade-Berechnung
hillshade_dem, hillshade_transform, hillshade_crs, hillshade_profile, hillshade_nodata = load_raster(hillshade_dem_path)


# Hillshade für das unabhängige DEM berechnen
hillshade_array = hillshade(hillshade_dem)

# Reprojizieren der Polyline, falls CRS nicht übereinstimmt
if profile:
    if polyline.crs != masked_dem1_transform:
        polyline = polyline.to_crs(dem1_crs)


if glacier_outline: 
    glacier_outline_loaded = gpd.read_file(glacier_outline_shape)
    if glacier_outline_loaded.crs != masked_dem1_transform:
        glacer_outline_loaded = glacier_outline_loaded.to_crs(dem1_crs)

# Bereiche außerhalb von dem_diff mit NaN versehen
dem_diff_masked = np.where(np.isnan(dem_diff), np.nan, dem_diff)





#########################################################

# Visualisierung: Differenz und Hillshade
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(1, 2, width_ratios=[20, 1], wspace=0.1)  # Passe die Breitenverhältnisse an


# Hintergrund-Hillshade über das gesamte Fenster
ax = fig.add_subplot(gs[0])
ax.imshow(hillshade_array, cmap='gray', alpha=0.5, extent=(
    hillshade_transform[2],
    hillshade_transform[2] + hillshade_transform[0] * hillshade_dem.shape[1],
    hillshade_transform[5] + hillshade_transform[4] * hillshade_dem.shape[0],
    hillshade_transform[5]
))  # Alpha-Wert für Transparenz


# Differenz darstellen
diff_img = ax.imshow(dem_diff_masked, cmap='RdYlBu_r', alpha=0.8, extent=(
    masked_dem1_transform[2],
    masked_dem1_transform[2] + masked_dem1_transform[0] * masked_dem1.shape[1],
    masked_dem1_transform[5] + masked_dem1_transform[4] * masked_dem1.shape[0],
    masked_dem1_transform[5]
))

# Polyline hinzufügen
if profile:
    for _, row in polyline.iterrows():
        x, y = row.geometry.xy
        plt.plot(x, y, color='black', linewidth=1, linestyle='dashed', label='Polyline')  # Farbe und Breite der Linie anpassen


if glacier_outline: 
     for _, row in glacier_outline_loaded.iterrows():
        if row.geometry.geom_type == 'Polygon':
            x, y = row.geometry.exterior.xy  # Äußere Begrenzung des Polygons
            plt.plot(x, y, color='blue', linewidth=1)  # Farbe und Breite der Linie anpassen
        elif row.geometry.geom_type == 'MultiPolygon':
            for polygon in row.geometry:
                x, y = polygon.exterior.xy
                plt.plot(x, y, color='blue', linewidth=1)
        else:
            print(f"Geometrietyp {row.geometry.geom_type} wird nicht unterstützt.")

ax.set_title('DoD ' + year1 + '/' + year2)

# Farbbalken und Titel hinzufügen
cbar = add_colorbar(diff_img)
cbar.set_label('Elevation Change (m)', rotation=90, labelpad=15)

# Optional: Farbbereich anpassen
diff_img.set_clim(-50, 50)


# Achsen auf die Ausdehnung des Differenzmodells begrenzen
ax.set_xlim(masked_dem1_transform[2], masked_dem1_transform[2] + masked_dem1_transform[0] * masked_dem1.shape[1])
ax.set_ylim(masked_dem1_transform[5] + masked_dem1_transform[4] * masked_dem1.shape[0], masked_dem1_transform[5])


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


