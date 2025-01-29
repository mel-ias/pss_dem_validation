import os, sys, glob
import numpy as np
import argparse

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


def getparser():
   
    # Create an ArgumentParser object with a description of the Planet4Stereo tool.
    # The ArgumentDefaultsHelpFormatter will display default values in the help message.
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--path_pss_dem_mosaic', help='Provide the path to the calculated and mosaiced PSS DEM to be validated', required=True)
    parser.add_argument('--path_validation_dem', help='Path to the DEM against which the PSS DEM should be compared', required=True) 
    parser.add_argument('--min_elevation', help='Provide expected minimum elevation of the PSS DEM, required for plot reasons', type=float, required=True)
    parser.add_argument('--max_elevation', help='Provide expected maximum elevation of the PSS DEM, required for plot reasons', type=float, required=True)

    # optional
    parser.add_argument('--path_shp_aoi', help = 'Path to the shape file that represents the area of interest of the PSS DEM', required=False)
    parser.add_argument('--path_shp_aoc', help = 'Path to the shape file that represents the area of comparison where the PSS DEM and the validation DEM are compared', required=False)
    parser.add_argument('--year_of_pss_dem', help='Provide the year of the PSS DEM to be added as prefix in the headlines of the plots (DEM and Dod)', default ="", required=False)
    parser.add_argument('--path_shp_contour_lines', help = 'Path to a shape file that shows contour lines, e.g. of glacier regions, optional if required in plot', required=False)
    return parser

def calculate_utm_zone(longitude, latitude):
    """
    Calculate UTM zone and EPSG code based on geographic lon/lat 
    """
    zone_number = int((longitude + 180) // 6) + 1
    is_northern = latitude >= 0
    epsg_code = 32600 + zone_number if is_northern else 32700 + zone_number
    return epsg_code


def reproject_to_utm(input_file, output_file):
    """
    project geographic coordinates to UTM 
    """
    with rasterio.open(input_file) as src:
        # get geo coordinates of the input DEM
        bounds = src.bounds
        centroid_longitude = (bounds.left + bounds.right) / 2
        centroid_latitude = (bounds.top + bounds.bottom) / 2
        
        # calculate epsg code / utm zone
        utm_epsg = calculate_utm_zone(centroid_longitude, centroid_latitude)
        utm_crs = CRS.from_epsg(utm_epsg)
        
        # calculate transformation parameters
        transform, width, height = calculate_default_transform(
            src.crs, utm_crs, src.width, src.height, *src.bounds
        )
        
        # extract metadata
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': utm_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # project and save
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
        print(f"Projected to UTM (EPSG:{utm_epsg}) and saved to {output_file}")
    return utm_epsg



def main (args):
    # Initialize the arugment parser and parse command line arguments
    parser = getparser()
    args = parser.parse_args(args)


    # Load required settings and input files
    path_pss_dem_mosaic = args.path_pss_dem_mosaic 
    path_validation_dem = args.path_validation_dem
    year_of_pss_dem = args.year_of_pss_dem
    min_elevation = args.min_elevation
    max_elevation = args.max_elevation
    path_shp_aoi = args.path_shp_aoi   
    path_shp_aoc = args.path_shp_aoc
    path_shp_contour_lines = args.path_shp_contour_lines


    # if a specfic area of interest is defined
    if path_shp_aoi:
        planetscope_dem_clipped = os.path.splitext(path_pss_dem_mosaic)[0] + "_clipped.tif"
        validation_dem_clipped = os.path.splitext(path_validation_dem)[0] + "_clipped.tif"
        rproc.clip_raster_by_shapefile(path_shp_aoi, path_pss_dem_mosaic, planetscope_dem_clipped)
        rproc.clip_raster_by_shapefile(path_shp_aoi, path_validation_dem, validation_dem_clipped)
        path_pss_dem_mosaic = planetscope_dem_clipped
        path_validation_dem = validation_dem_clipped



    # 1. demcoreg requires projected coordinates, output DEM of planet4stereo are geographic coordinates, i.e. reproject from geographic coordinates
    input_file =  path_pss_dem_mosaic
    output_file = path_pss_dem_mosaic + "_utm.tif"
    utm_epsg = reproject_to_utm(input_file, output_file)
    path_pss_dem_mosaic = output_file

    input_file =  path_validation_dem
    output_file = path_validation_dem + "_utm.tif"
    reproject_to_utm(input_file, output_file)
    path_validation_dem = output_file


    # 2. perform co-registration for validation using dem_coreg 
    cmd_coreg = [
        '-outdir',  os.path.dirname(path_pss_dem_mosaic),
        '-mode', 'nuth',
        '-mask_list', 'glaciers',
        '-res', 'mean',
        path_validation_dem, path_pss_dem_mosaic]
    dem_align.main(cmd_coreg)
    out_dem_aligned_mosaic = glob.glob(os.path.splitext(path_pss_dem_mosaic)[0] + '*_align.tif' )[0]    

    # 3. Compute differences between referende DEM and generated (non-gap-filled) DEM in stable areas and provide statistics.
    out_dem_aligned_mosaic_resamp = os.path.splitext(out_dem_aligned_mosaic)[0] + "_resamp.tif"
    out_dem_aligned_mosaic_resamp_roi = os.path.splitext(out_dem_aligned_mosaic)[0] + "_resamp_roi.tif"
    refdem_resamp = os.path.splitext(path_validation_dem)[0] + "_resamp.tif"
    refdem_resamp_roi = os.path.splitext(path_validation_dem)[0] + "_resamp_roi.tif"
    out_dod = os.path.splitext(out_dem_aligned_mosaic)[0] + "_resamp_roi_diff.tif"

    # resample planet DEM to same resolution as reference DEM
    # a) get resolution of reference DEM
    with rasterio.open(path_validation_dem) as ds:
        gt = ds.transform
        xres = np.rint(gt[0])
        yres = np.rint(-gt[4])  
    # b) run resampling of planet DEM / ref DEM to avoid rounding errors in pixel spacing 
    rproc.resample_res(out_dem_aligned_mosaic, out_dem_aligned_mosaic_resamp, xres, yres)
    rproc.resample_res(path_validation_dem, refdem_resamp, xres, yres)

    # clip planet DEM and reference DEM to ROI
    # no ROI given? calculate overlapping area, create shape and use this as clipping mask
    if not path_shp_aoc: 
        path_shp_aoc = os.path.splitext(refdem_resamp)[0] + "_overlap_pssdem.shp"
        epsg_code = 'epsg:' + utm_epsg
        rproc.save_overlap_as_shapefile(out_dem_aligned_mosaic_resamp, refdem_resamp, path_shp_aoc, epsg_code=epsg_code) # 32719
    rproc.clip_raster_by_shapefile(path_shp_aoc, out_dem_aligned_mosaic_resamp, out_dem_aligned_mosaic_resamp_roi)
    rproc.clip_raster_by_shapefile(path_shp_aoc, refdem_resamp, refdem_resamp_roi)

    # clip planet DEM and reference DEM to same extent to perform DoD calculation (reference DEM should cover planet DEM)
    rproc.clip_2rasters_1extent(refdem_resamp_roi, out_dem_aligned_mosaic_resamp_roi, refdem_resamp_roi, out_dem_aligned_mosaic_resamp_roi )

    # calc DoD, output stats
    stats, dod = rproc.calc_dod(refdem_resamp_roi, out_dem_aligned_mosaic_resamp_roi, out_dod, out_dem_aligned_mosaic, path_shp_contour_lines, min_elevation, max_elevation, visu=True, year = year_of_pss_dem, output_path_print=os.path.splitext(out_dem_aligned_mosaic)[0] + "_dod.png")

    # print PSS DEM figure:
    print ("min elevation", min_elevation)
    print ("max elevation", max_elevation)
    rproc.print_dem(out_dem_aligned_mosaic, os.path.splitext(out_dem_aligned_mosaic)[0] + "_dem.png", min_elevation = min_elevation, max_elevation = max_elevation, year = year_of_pss_dem, path_shape_file= path_shp_contour_lines)

    # print statistics
    print ("validation, stats:")
    for attribute, value in stats.items():
        print('{} : {:.2f}'.format(attribute, value))



if __name__=="__main__":
    main(sys.argv[1:])
