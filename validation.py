import os, glob
import numpy as np

# GIS libraries
import rasterio

# Multicore processing
import psutil

# Get the number of available CPU cores
n_cpu = psutil.cpu_count(logical=False)

# Planet4Stereo modules
import planet_raster_processing as rproc


from demcoreg import dem_align

# # 2019 Shisper
planetscope_dem = "/home/mela/samples/wsl_ames_test_shi_2019/PSScene_overlap_10_converg_4.0_map/final_rpc_stereo/mosaic_dems/point_cloud_mosaic-DEM.tif"
validation_dem = "/home/mela/samples/wsl_ames_test_shi_2019/PSScene_overlap_10_converg_4.0_map/refdem/output_COP30.tif-adj.tif_utm.tif"
epsg_code = "EPSG:32643"
min_elevation = 0.0
max_elevation = 7000.0
in_compare_mask = "/mnt/n/DATEN/KARAKORAM_SHISPER/Shisper_StableArea_ShapeFile/Shi_AliAbad_stable.shp"
year = str(2019)

# 2017 Shisper 
# planetscope_dem = "/home/mela/samples/wsl_ames_test_shi_2017/PSScene_overlap_10_converg_4.0_map/final_rpc_stereo/mosaic_dems/point_cloud_mosaic-DEM.tif"
# validation_dem = "/home/mela/samples/wsl_ames_test_shi_2017/PSScene_overlap_10_converg_4.0_map/refdem/output_COP30.tif-adj.tif_utm.tif"
# epsg_code = "EPSG:32643"
# min_elevation = 0.0
# max_elevation = 7000.0
# in_compare_mask = "/mnt/n/DATEN/KARAKORAM_SHISPER/Shisper_StableArea_ShapeFile/Shi_AliAbad_stable.shp"
# year = str(2017)

#2019 Boverbrean
# planetscope_dem  = "/home/mela/samples/wsl_ames_test_bov_2021/202109_PSS4_overlap_10_converg_4.0_map/final_rpc_stereo/mosaic_dems/point_cloud_mosaic-DEM.tif"
# validation_dem = "/home/mela/samples/change_detection_bov/ndh_jostedalsbreen_1pkt/ndh_jostedalsbreen_2pkt_wgs84_1m.tif"
# epsg_code = 'EPSG:32632'
# min_elevation = 0.0
# max_elevation = 3000.0
# in_compare_mask = "" #"/home/mela/samples/change_detection_bov/ndh_Jostalsbreen_no_glac/ndh_no_glac.shp"
# year = str(2021)


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
    rproc.save_overlap_as_shapefile(out_dem_aligned_mosaic_resamp, refdem_resamp, in_compare_mask, epsg_code=epsg_code) # 32719
rproc.clip_raster_by_shapefile(in_compare_mask, out_dem_aligned_mosaic_resamp, out_dem_aligned_mosaic_resamp_roi)
rproc.clip_raster_by_shapefile(in_compare_mask, refdem_resamp, refdem_resamp_roi)

# 3. clip planet DEM and reference DEM to same extent to perform DoD calculation (reference DEM should cover planet DEM)
rproc.clip_2rasters_1extent(refdem_resamp_roi, out_dem_aligned_mosaic_resamp_roi, refdem_resamp_roi, out_dem_aligned_mosaic_resamp_roi )

# 4. calc DoD, output stats
stats, dod = rproc.calc_dod(refdem_resamp_roi, out_dem_aligned_mosaic_resamp_roi, out_dod, visu=True, year = year, output_path_print=os.path.splitext(out_dem_aligned_mosaic)[0] + "_dod.png")

# 5. print pss DEM figure:
print ("min elevation", min_elevation)
print ("max elevation", max_elevation)
rproc.print_dem(out_dem_aligned_mosaic, os.path.splitext(out_dem_aligned_mosaic)[0] + "_dem.png", min_elevation = min_elevation, max_elevation = max_elevation, year = year)


print ("validation, stats:")
for attribute, value in stats.items():
    print('{} : {:.2f}'.format(attribute, value))


