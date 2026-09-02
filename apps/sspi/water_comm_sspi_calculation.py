"""
water_comm - tool to compute SSPI
__version__ = '1.0.0'
Version(s):
(1.0.0) --> First release
"""
# -------------------------------------------------------------------------------------
# Complete library
import logging
from os.path import join
from argparse import ArgumentParser
import os
from time import time, strftime, gmtime
import pandas
import numpy as np
from scipy import stats
import rasterio
from rasterio.features import geometry_mask
import matplotlib.pyplot as plt
import geopandas as gpd
import calendar 
from collections import defaultdict
from src.json_utilities.water_comm_utilities_json import read_file_json
from src.time_utilities.water_comm_utilities_time import set_time_new
from src.geo_utilities.water_comm_utilities_geo import read_file_raster

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# Algorithm information
alg_project = 'water_comm'
alg_name = 'SSPI CALCULATION'
alg_version = '1.0.0'
alg_release = '2025-05-13'
alg_type = 'SSPICALCULATION'
# Algorithm parameter(s)
time_format_algorithm = '%Y-%m-%d %H:%M'
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

# Script Main
def main():
    # Get algorithm settings
    [file_script, file_settings, time_arg] = get_args()

    # Set algorithm settings
    data_settings = read_file_json(file_settings)

    # Set algorithm logging
    os.makedirs(data_settings['data']['log_folder'], exist_ok=True)
    set_logging(logger_file=join(data_settings['data']['log_folder'], data_settings['data']['log_filename']))
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
    # Info algorithm
    logging.info('[' + alg_project + ' ' + alg_type + ' - ' + alg_name + ' (Version ' + alg_version + ')]')
    logging.info('[' + alg_project + '] Execution Time: ' + strftime("%Y-%m-%d %H:%M", gmtime()) + ' GMT')
    logging.info('[' + alg_project + '] Reference Time: ' + time_arg + ' GMT')
    logging.info('[' + alg_project + '] Start Program ... ')
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
    # Time algorithm information
    start_time = time()

    logging.info( ' Setting time run...')
    # Organize time run
    time_run, full_time_range, time_chunks = set_time_new(
        time_run_args=time_arg,
        time_run_file=data_settings['time']['time_run'],
        time_run_file_start=data_settings['time']['time_start'],
        time_run_file_end=data_settings['time']['time_end'],
        time_format=time_format_algorithm,
        time_period=data_settings['time']['time_period'],
        time_frequency=data_settings['time']['time_frequency'],
        time_rounding=data_settings['time']['time_rounding'],
        time_reverse=True)

    logging.info(f"Time range for SSPI calculation: {full_time_range}")
    logging.info(' Setting time run...DONE')
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
    # Load areacell
    logging.info(' --> Load target areacell ... ')
    da_areacell, wide_areacell, high_areacell, proj_areacell, transform_areacell, \
        bounding_box_areacell, no_data_areacell, crs_areacell, lons_areacell, lats_areacell = \
        read_file_raster(data_settings['data']['inputs']['areacell'],
                         coord_name_x='lon', coord_name_y='lat',
                         dim_name_x='lon', dim_name_y='lat')
    logging.info(' --> Load target areacell ... DONE')
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
    logging.info(' --> Load Inputs ... ')
    gamma_csv = data_settings["data"]["inputs"]["gamma_values"]
    winter_months = data_settings["data"]["inputs"]["winter_months"]
    flag_refit = data_settings["algorithm"]["flags"]["flag_refit"]
    flag_10 = data_settings["algorithm"]["flags"]["flag_10"]
    basin_folder = data_settings["data"]["inputs"]["basins_shp_list"]
    realtime_folder = data_settings["data"]["inputs"]["realtime_folder"]
    reanalysis_folder = data_settings["data"]["inputs"]["reanalysis_folder"]
    basins_combined_shp_path = data_settings["data"]["inputs"]["basins_combined_shp"]
    output_file_path = data_settings["data"]["outcome"]["csv_path"]
    output_folder = data_settings["data"]["outcome"]["folder"]
    monthly_maps_folder = data_settings["data"]["outcome"]["monthly_maps_folder"]

    os.makedirs(monthly_maps_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    ks_stat = np.nan

    if flag_refit==1:
        gamma_df= None
    else :
        gamma_df = pandas.read_csv(gamma_csv)
        
    basins_combined_shp = gpd.read_file(basins_combined_shp_path)
    basins_combined_shp["basin_name"] = (basins_combined_shp["NOME"].astype(str).str.replace(r"\s*-\s*", "_", regex=True) .str.replace(" ", "_", regex=False))
    map_bounds = basins_combined_shp.total_bounds  # [minx, miny, maxx, maxy]

    logging.info(' --> Load Inputs ... DONE ')
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
    # Get basin list from the folder path.
    shapefiles = [f for f in os.listdir(basin_folder) if f.endswith(".shp")]
    basins = {}

    for shp_file in shapefiles:

        full_path = os.path.join(basin_folder, shp_file)
        gdf = gpd.read_file(full_path)

        name_no_ext = os.path.splitext(shp_file)[0]

        parts = name_no_ext.split("_", 2)
        if len(parts) >= 3:
            basin_id = int(parts[1])
            basin_name = parts[2]
        else:
            basin_id = -1
            basin_name = name_no_ext

        basins[basin_name] = gdf
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
    # CREATE TIME SERIES MASKED WITH BASINS SHAPE
    logging.info(' --> Creating timeseries ...')
    
    for time_date in full_time_range:
        
        results = defaultdict(list)
        all_sspi = []
            
        if flag_refit == 1:
            start_date = data_settings['time']['time_start_refit']
            end_date = data_settings['time']['time_end_refit']
            time_range = pandas.date_range(start=start_date, end=end_date, freq='D')

        elif flag_10 == 1:
            period = get_period(time_date)
            time_range = pandas.date_range( start=time_date - pandas.Timedelta(days=30),end=time_date,freq='D')

        else:
            time_range = pandas.date_range(start=time_date.replace(day=1),end=time_date,freq='D')

            
        for  current_date in time_range:
            
            date_str = current_date.strftime('%Y%m%d')
            folder_reanalysis = os.path.join(reanalysis_folder, current_date.strftime("%Y"), current_date.strftime("%m"), current_date.strftime("%d"))
            folder_realtime = os.path.join(realtime_folder, current_date.strftime("%Y"), current_date.strftime("%m"), current_date.strftime("%d"))
            swe_file = f"ITSNOW500-SWE_{date_str}110000.tif"
            swe_path = None

            if os.path.exists(os.path.join(folder_reanalysis, swe_file)):
                swe_path = os.path.join(folder_reanalysis, swe_file)
            elif os.path.exists(os.path.join(folder_realtime, swe_file)):
                swe_path = os.path.join(folder_realtime, swe_file)

            if swe_path is None:
                logging.warning(f"Missing SWE file for {time_date}")
                for basin in basins:
                    results[basin].append({"date": time_date, "SWE": np.nan})
                continue

            try:
                with rasterio.open(swe_path) as swe_src:
                    swe = swe_src.read(1).astype(float)
                    nodata_swe = swe_src.nodata
                    if nodata_swe is not None:
                        swe[swe == nodata_swe] = 0.0
                    swe = (swe / 1000) * da_areacell  # mm → Mm³

                    # Mask SWE by each basin
                    for basin_name, gdf in basins.items():

                        geoms = [geom for geom in gdf.geometry if geom is not None]

                        mask_arr = geometry_mask(
                            geoms,
                            transform=swe_src.transform,
                            invert=True,
                            out_shape=swe.shape
                        )

                        swe_masked = np.where(mask_arr, swe, 0.0)
                        swe_sum = np.nansum(swe_masked)

                        results[basin_name].append({
                            "date": current_date,
                            "SWE": swe_sum
                        
                        })


                logging.info(f"Processed SWE for {current_date}")

            except Exception as e:
                logging.error(f"Error processing {current_date}: {e}")
                for basin in basins.items():
                    results[basin].append({"date": current_date, "SWE": np.nan})
        logging.info(' --> Creating timeseries ...done')
        
        # -------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------
        # Compute mean SWE over a rolling window of 30 days for each basin
        df = pandas.concat([pandas.DataFrame(v).assign(basin_name=k) for k, v in results.items()])
        df = df.pivot(index="date", columns="basin_name", values="SWE").sort_index()
        df_roll = df.mean().to_frame(name="SWE_Mm3")
        # Add month and year columns for grouping
        month = time_date.month
        year = time_date.year
        period = get_period(time_date) if flag_10 == 1 else None
       
        # -------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------
        # CALCULATE SSPI
        logging.info(' --> Calculating SSPI ...')
        
        for basin in df_roll.index:
            sub_series = df_roll.loc[basin, "SWE_Mm3"]
            tmp = process(sub_series,basin, month, period, gamma_df, flag_refit,flag_10)
            if tmp is not None:
                all_sspi.append(tmp)
            else:
                logging.warning(f"Insufficient data for SSPI calculation for basin {basin} in {year}-{month:02d} period {period}")
                all_sspi.append(pandas.DataFrame({
                    "basin_name": [basin],
                    "SSPI": [np.nan],
                    "SWE_Mm3": [sub_series],
                    "month": [month],
                    "shape": [np.nan],
                    "loc": [np.nan],
                    "scale": [np.nan],
                    "p0": [np.nan],
                    "ks_pvalue": [np.nan],
                    "period": [period] if flag_10 == 1 else [np.nan]
                }))
    
        logging.info(' --> Calculating SSPI ...DONE')
        # -------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------
        # SAVE SSPI
        sspi_df = pandas.concat(all_sspi)
        sspi_df.sort_index(inplace=True)
        path = os.path.join(output_file_path, f"SSPI_{year}_{month:02d}_{period}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sspi_df.to_csv(path, index=True)
        logging.info(f"SSPI saved to {output_file_path}")
        # -------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------
        # PLOT MONTHLY MAPS FOR EACH YEAR
        logging.info(' --> Plotting SSPI ...')
        vmin_global = -2
        vmax_global = 2
        cmap = 'RdBu'

        sspi_df["basin_name"] = sspi_df["basin_name"].str.replace("AdBAlpiOR_", "",regex=False)
        
        #sspi_df = sspi_df.T

        # merge shapefile with SSPI data
        if flag_10 ==1:
            basin_shp_selected = (basins_combined_shp.set_index("basin_name").join(
                    sspi_df.set_index("basin_name")[
                    ["SSPI", "SWE_Mm3","period","month" ,"shape", "loc", "scale", "p0", "ks_pvalue"]])
                .dropna(subset=["SSPI"])
                .reset_index())           
            output_shp = os.path.join(output_folder, f"SSPI_{year}_{month:02d}_{period}.shp")
        else: 
            basin_shp_selected = (basins_combined_shp.set_index("basin_name").join(
                    sspi_df.set_index("basin_name")[[
                        "SSPI", "SWE_Mm3","month", "shape", "loc", "scale", "p0", "ks_pvalue"]])
                .dropna(subset=["SSPI"])
                .reset_index())   
            output_shp = os.path.join(output_folder, f"SSPI_{year}_{month:02d}.shp")

        basin_shp_selected.to_file(output_shp)

        x_range = map_bounds[2] - map_bounds[0]
        y_range = map_bounds[3] - map_bounds[1]
        fig_width = 4

        fig_height = fig_width * (y_range / x_range)  # keep aspect ratio
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        basin_shp_selected.plot(
            column='SSPI',
            ax=ax,
            cmap=cmap,
            vmin=vmin_global,
            vmax=vmax_global,
            edgecolor='black',
            linewidth=0.5,
            missing_kwds={"color": "lightgrey", "label": "No data"}
        )

        # consistent extent
        ax.set_xlim(map_bounds[0], map_bounds[2])
        ax.set_ylim(map_bounds[1], map_bounds[3])
        ax.set_aspect('equal')
        ax.set_axis_off()

        # colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin_global, vmax=vmax_global))
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
        cbar.set_label("SSPI", fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        
        if flag_10 == 1:
            ax.set_title(f"SSPI {year} - {month:02d} - {period}", fontsize=15)
            output_png = os.path.join(monthly_maps_folder, f"SSPI_{year}_{month:02d}_{period}.png")
        else :
            ax.set_title(f"SSPI {year} - {month:02d} ", fontsize=15)
            output_png = os.path.join(monthly_maps_folder, f"SSPI_{year}_{month:02d}.png")
            
            
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close(fig)

    logging.info(' --> Plotting SSPI ...DONE')
    return None
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# Method to get script argument(s)
def get_args():

    parser_handle = ArgumentParser()
    parser_handle.add_argument('-settings_file', action="store", dest="alg_settings")
    parser_handle.add_argument('-time_now', action="store", dest="alg_time_now")
    parser_values = parser_handle.parse_args()

    alg_script = parser_handle.prog

    if parser_values.alg_settings:
        alg_settings = parser_values.alg_settings
    else:
        alg_settings = 'configuration.json'

    if parser_values.alg_time_now:
        alg_time_now = parser_values.alg_time_now
    else:
        alg_time_now = None

    return alg_script, alg_settings, alg_time_now

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# Method to set logging information
def set_logging(logger_file='log.txt', logger_format=None):
    if logger_format is None:
        logger_format = '%(asctime)s %(name)-12s %(levelname)-8s ' \
                        '%(filename)s:[%(lineno)-6s - %(funcName)20s()] %(message)s'

    # Remove old logging file
    if os.path.exists(logger_file):
        os.remove(logger_file)

    # Set level of root debugger
    logging.root.setLevel(logging.INFO)

    # Open logging basic configuration
    logging.basicConfig(level=logging.INFO, format=logger_format, filename=logger_file, filemode='w')

    # Set logger handle
    logger_handle_1 = logging.FileHandler(logger_file, 'w')
    logger_handle_2 = logging.StreamHandler()
    # Set logger level
    logger_handle_1.setLevel(logging.INFO)
    logger_handle_2.setLevel(logging.INFO)
    # Set logger formatter
    logger_formatter = logging.Formatter(logger_format)
    logger_handle_1.setFormatter(logger_formatter)
    logger_handle_2.setFormatter(logger_formatter)
    # Add handle to logging
    logging.getLogger('').addHandler(logger_handle_1)
    logging.getLogger('').addHandler(logger_handle_2)
# ------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
#Method to get period
def get_period(date):
    day = date.day
    if 1 <= day <= 10:
        return 1
    elif 11 <= day <= 20:
        return 2
    else:
        return 3
    
# ------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
#Method to calculate sspi
def calculate_sspi(x, shape, loc, scale, p0):
    x = np.asarray(x, dtype=float)

    gamma_cdf = stats.gamma.cdf(np.where(x <= 0, 1e-6, x),
                                shape, loc=loc, scale=scale)
    cdf = p0 + (1 - p0) * gamma_cdf
    cdf = np.clip(cdf, 1e-6, 1 - 1e-6)

    return stats.norm.ppf(cdf)
# ------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# Method to subdivide periods
def process(subset, basin, month, val, gamma_df, flag_refit, flag_10):

    if flag_refit == 0:
        
        if flag_10 == 1:
            gamma_row = gamma_df[(gamma_df["basin_name"] == basin) &(gamma_df["month"] == month) & (gamma_df["period"] == val)]
        else:
            gamma_row = gamma_df[(gamma_df["basin_name"] == basin) & (gamma_df["month"] == month)]
        
        if gamma_row.empty:
            return None
        
        x = subset/ 1e6 # in this case i have only a float value
        shape = gamma_row["shape"].iloc[0]
        loc = gamma_row["loc"].iloc[0]
        scale = gamma_row["scale"].iloc[0]
        p0 = gamma_row["p0"].iloc[0]
        ks_stat = np.nan
        sspi = float(calculate_sspi(x, shape, loc, scale, p0))
        swe_out = float(x)

    else:
        subset_valid = subset.copy().dropna() / 1e6

        if len(subset_valid) < 10:
            return None
        
        x = subset_valid.values
        
        x_pos = x[x > 0]

        if len(x_pos) < 5:
            return None
        
        p0 = np.sum(x <= 0) / len(x)
        shape, loc, scale = stats.gamma.fit(x_pos, floc=0)
        ks_stat, p_value = stats.kstest( x_pos, "gamma", args=(shape, loc, scale))

        if p_value < 0.05:
            return None

        sspi = calculate_sspi(x, shape, loc, scale, p0)
        swe_out = subset_valid.values

    tmp = pandas.DataFrame({
    "basin_name": [basin],
    "SSPI": [sspi],
    "SWE_Mm3": [swe_out],
    "month": [month],
    "shape": [shape],
    "loc": [loc],
    "scale": [scale],
    "p0": [p0],
    "ks_pvalue": [ks_stat],
    })

    if flag_10 == 1:
        tmp["period"] = val

    return tmp
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Call script from external library
if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------
