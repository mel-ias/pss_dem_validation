import os, glob
import numpy as np

# GIS libraries
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import CRS

# Multicore processing
import psutil

# Get the number of available CPU cores
n_cpu = psutil.cpu_count(logical=False)

# Planet4Stereo modules
import planet_raster_processing as rproc
from demcoreg import dem_align



# # 2017 Shisper - noutm
# planetscope_dem = "/home/mela/samples/wsl_ames_test_shi_2019_ellips/PSScene_overlap_10_converg_4.0_map/final_rpc_stereo/mosaic_dems/point_cloud_mosaic-DEM.tif"
# validation_dem = "/home/mela/samples/wsl_ames_test_shi_2019_ellips/PSScene_overlap_10_converg_4.0_map/refdem/output_COP30.tif-adj.tif"
# year = str(2019)
# min_elevation = 0.0
# max_elevation = 7000.0
# aoi = "/home/mela/samples/change_detection_shisper/glacier_mask/aoi_shisper.shp"
# in_compare_mask = "/mnt/n/DATEN/KARAKORAM_SHISPER/Shisper_StableArea_ShapeFile/Shi_AliAbad_stable.shp"
# glacier_shape = "/home/mela/samples/change_detection_shisper/glacier_mask/shisper_glacier_mask.shp"
# ## shisper: gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.0], height_ratios = [2.5, 1] ) 
# Achtung! Breite in planet_raster_processing.py wieder auf alten Wert zurückstellen und ggf. die datalim s rausnehmen um alten Plot-Stil zu bekommen

# 2021 Boverbrean
planetscope_dem = "/home/mela/samples/wsl_ames_test_bov_2021_ellips_2/202109_PSS4_overlap_10_converg_4.0_map/final_rpc_stereo/mosaic_dems/point_cloud_mosaic-DEM.tif"
#validation_dem = "/home/mela/samples/change_detection_bov/ndh_jostedalsbreen_1pkt/ndh_2pkt_wgs84_ellips.tif"
validation_dem = "/home/mela/samples/wsl_ames_test_bov_2021_ellips_2/202109_PSS4_overlap_10_converg_4.0_map/refdem/output_COP30.tif-adj.tif"
year = str(2021)
min_elevation = 1000.0
max_elevation = 2500.0
aoi =  "/home/mela/samples/change_detection_bov/glaciers_jotunheimen_mask/outline_roi_ndh_planet.shp" 
in_compare_mask =  "/home/mela/samples/change_detection_bov/glaciers_jotunheimen_mask/outline_roi_ndh_planet_no_glac.shp" 
glacier_shape = "/home/mela/samples/change_detection_bov/glaciers_jotunheimen_mask/glacier_mask_jotunheimen.shp"

## bov fig verhältnisse: 
#gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], height_ratios = [1.25, 1] ) 


def calculate_utm_zone(longitude, latitude):
    """
    Berechnet die UTM-Zone und den EPSG-Code basierend auf Länge und Breite.
    """
    zone_number = int((longitude + 180) // 6) + 1
    is_northern = latitude >= 0
    epsg_code = 32600 + zone_number if is_northern else 32700 + zone_number
    return epsg_code


def reproject_to_utm(input_file, output_file):
    """
    Reprojiziert ein GeoTIFF von WGS84 in die entsprechende UTM-Projektion.
    """
    with rasterio.open(input_file) as src:
        # Extrahiere die Geokoordinaten des Rasters
        bounds = src.bounds
        centroid_longitude = (bounds.left + bounds.right) / 2
        centroid_latitude = (bounds.top + bounds.bottom) / 2
        
        # Berechne den EPSG-Code für die UTM-Zone
        utm_epsg = calculate_utm_zone(centroid_longitude, centroid_latitude)
        utm_crs = CRS.from_epsg(utm_epsg)
        
        # Berechne die Transformationsparameter
        transform, width, height = calculate_default_transform(
            src.crs, utm_crs, src.width, src.height, *src.bounds
        )
        
        # Metadaten aktualisieren
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': utm_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Reprojektion und Speichern
        with rasterio.open(output_file, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=utm_crs,
                    resampling=Resampling.cubic
                )
        print(f"Reprojected to UTM (EPSG:{utm_epsg}) and saved to {output_file}")
    return utm_epsg



if aoi:
    planetscope_dem_clipped = os.path.splitext(planetscope_dem)[0] + "_clipped.tif"
    validation_dem_clipped = os.path.splitext(validation_dem)[0] + "_clipped.tif"
    #intersection_err_dem_clipped = os.path.splitext(intersection_err_dem)[0] + "_clipped.tif"
    rproc.clip_raster_by_shapefile(aoi, planetscope_dem, planetscope_dem_clipped)
    rproc.clip_raster_by_shapefile(aoi, validation_dem, validation_dem_clipped)
    #rproc.clip_raster_by_shapefile(aoi, intersection_err_dem, intersection_err_dem_clipped)
    planetscope_dem = planetscope_dem_clipped
    validation_dem = validation_dem_clipped


# reproject from geographic coordinates
input_file =  planetscope_dem
output_file = planetscope_dem + "_utm.tif"
utm_epsg = reproject_to_utm(input_file, output_file)
planetscope_dem = output_file

input_file =  validation_dem
output_file = validation_dem + "_utm.tif"
reproject_to_utm(input_file, output_file)
validation_dem = output_file


# 1. perform co-registration to validation dem using dem_coreg 
cmd_coreg = [
    '-outdir',  os.path.dirname(planetscope_dem),
    '-mode', 'nuth',
    '-mask_list', 'glaciers',
    '-res', 'mean',
    validation_dem, planetscope_dem]
dem_align.main(cmd_coreg)


out_dem_aligned_mosaic = glob.glob(os.path.splitext(planetscope_dem)[0] + '*_align.tif' )[0]
print (out_dem_aligned_mosaic)    
   

##################
### VALIDATION ###
##################
# Compute differences between referende DEM and generated (non-gap-filled) DEM in stable areas and provide statistics.
out_dem_aligned_mosaic_resamp = os.path.splitext(out_dem_aligned_mosaic)[0] + "_resamp.tif"
out_dem_aligned_mosaic_resamp_roi = os.path.splitext(out_dem_aligned_mosaic)[0] + "_resamp_roi.tif"
refdem_resamp = os.path.splitext(validation_dem)[0] + "_resamp.tif"
refdem_resamp_roi = os.path.splitext(validation_dem)[0] + "_resamp_roi.tif"
out_dod = os.path.splitext(out_dem_aligned_mosaic)[0] + "_resamp_roi_diff.tif"

# 1. resample planet DEM to same resolution as reference DEM
# a) get resolution of reference DEM
with rasterio.open(validation_dem) as ds:
    gt = ds.transform
    xres = np.rint(gt[0])
    yres = np.rint(-gt[4])  
# b) run resampling of planet DEM / ref DEM to avoid rounding errors in pixel spacing 
rproc.resample_res(out_dem_aligned_mosaic, out_dem_aligned_mosaic_resamp, xres, yres)
rproc.resample_res(validation_dem, refdem_resamp, xres, yres)

# 2. clip planet DEM and reference DEM to ROI
# no ROI given? calculate overlapping area, create shape and use this as clipping mask
if not in_compare_mask: 
    in_compare_mask = os.path.splitext(refdem_resamp)[0] + "_overlap_pssdem.shp"
    epsg_code = 'epsg:' + utm_epsg
    rproc.save_overlap_as_shapefile(out_dem_aligned_mosaic_resamp, refdem_resamp, in_compare_mask, epsg_code=epsg_code) # 32719
rproc.clip_raster_by_shapefile(in_compare_mask, out_dem_aligned_mosaic_resamp, out_dem_aligned_mosaic_resamp_roi)
rproc.clip_raster_by_shapefile(in_compare_mask, refdem_resamp, refdem_resamp_roi)

# 3. clip planet DEM and reference DEM to same extent to perform DoD calculation (reference DEM should cover planet DEM)
rproc.clip_2rasters_1extent(refdem_resamp_roi, out_dem_aligned_mosaic_resamp_roi, refdem_resamp_roi, out_dem_aligned_mosaic_resamp_roi )

# 4. calc DoD, output stats
stats, dod = rproc.calc_dod(refdem_resamp_roi, out_dem_aligned_mosaic_resamp_roi, out_dod, out_dem_aligned_mosaic, glacier_shape, min_elevation, max_elevation, visu=True, year = year, output_path_print=os.path.splitext(out_dem_aligned_mosaic)[0] + "_dod.png")

# 5. print pss DEM figure:
print ("min elevation", min_elevation)
print ("max elevation", max_elevation)
rproc.print_dem(out_dem_aligned_mosaic, os.path.splitext(out_dem_aligned_mosaic)[0] + "_dem.png", min_elevation = min_elevation, max_elevation = max_elevation, year = year, path_shape_file= glacier_shape)


print ("validation, stats:")
for attribute, value in stats.items():
    print('{} : {:.2f}'.format(attribute, value))


