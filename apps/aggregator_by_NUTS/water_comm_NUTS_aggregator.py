"""
water_comm - tool to summarize map by NUTS
__version__ = '1.0.0'
Version(s):
(1.0.0) --> First release
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Complete library
import logging
from os.path import join
from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
from time import time, strftime, gmtime
import pandas as pd
from datetime import datetime
import xarray as xr
import rioxarray  # Ensure rioxarray is imported
import numpy as np
import geopandas as gpd
np.set_printoptions(legacy='1.25')

from src.json_utilities.water_comm_utilities_json import read_file_json
from src.time_utilities.water_comm_utilities_time import set_time
from src.geo_utilities.water_comm_utilities_geo import read_file_raster
from src.generic_utilties.water_comm_generic_utilities import fill_tags2string
from src.geo_utilities.water_comm_utilities_read_SWE import load_SWE_map
from src.geo_utilities.water_comm_utilities_aggregator_by_NUTS import aggregator_by_NUTS
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Algorithm information
alg_project = 'water_comm'
alg_name = 'SUMMARIZE BY NUTS'
alg_version = '1.0.0'
alg_release = '2025-05-13'
alg_type = 'SUMMARIZEBYNUTS'
# Algorithm parameter(s)
time_format_algorithm = '%Y-%m-%d %H:%M'
# -------------------------------------------------------------------------------------

# Script Main
def main():

    # -------------------------------------------------------------------------------------
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

    # Organize time run
    time_run, time_range, time_chunks = set_time(
        time_run_args=time_arg,
        time_run_file=data_settings['time']['time_run'],
        time_run_file_start=data_settings['time']['time_start'],
        time_run_file_end=data_settings['time']['time_end'],
        time_format=time_format_algorithm,
        time_period=data_settings['time']['time_period'],
        time_frequency=data_settings['time']['time_frequency'],
        time_rounding=data_settings['time']['time_rounding'],
        time_reverse=True
    )
    # -------------------------------------------------------------------------------------
    # Load grid
    logging.info(' --> Load target grid ... ')
    da_domain_target, wide_domain_target, high_domain_target, proj_domain_target, transform_domain_target, \
        bounding_box_domain_target, no_data_domain_target, crs_domain_target, lons_target, lats_target = \
        read_file_raster(data_settings['data']['input_grid']['grid'],
                         coord_name_x='lon', coord_name_y='lat',
                         dim_name_x='lon', dim_name_y='lat')
    logging.info(' --> Load target grid ... DONE')
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Load NUTS from json
    logging.info(' --> Load NUTS ... ')
    nuts_gdf = gpd.read_file(data_settings['data']['input_grid']['nuts'])
    logging.info(' --> Load NUTS ... DONE')
    logging.info(' --> Number of NUTS regions: ' + str(nuts_gdf.shape[0]))
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Do the work
    for time_i, time_date in enumerate(time_range):

        # -------------------------------------------------------------------------------------
        # we immediately handle leap years by changing Feb 29 to Feb 28 just in case
        if time_date.month == 2 and time_date.day == 29:
            time_date = datetime(time_date.year, 2, 28, time_date.hour, time_date.minute)
            logging.warning('Leap year detected. Feb 29 changed to Feb 28')
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # we load the concerned map
        logging.info(' --> Load map for ' + str(time_date) + ' ... ')
        tags_template = data_settings['algorithm']['template']
        path_maps = data_settings['data']['input_list'][:]  # this is in principle a list of files
        negative_values_to_zero = data_settings['algorithm']['flags']['negative_values_to_zero']
        anomaly_map = load_SWE_map(time_date, high_domain_target, wide_domain_target,
                                     path_maps.copy(), tags_template, no_data_domain_target, da_domain_target,
                                   negative_values_to_zero=negative_values_to_zero)
        anomaly_map = np.squeeze(anomaly_map)
        logging.info(' --> Load map for ' + str(time_date) + ' ... DONE')
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # we compute statistics by NUTS
        logging.info('Compute statistics by NUTS ... ')
        anomaly_map_xr = xr.DataArray(anomaly_map, dims=['y', 'x'],
                                      coords={'y': lats_target[:,0], 'x': lons_target[0,:]})
        anomaly_map_xr = anomaly_map_xr.rio.write_crs(crs_domain_target)
        column_adm = data_settings['algorithm']['flags']['column_adm']
        stats_by_nuts = aggregator_by_NUTS(anomaly_map_xr, nuts_gdf, column_adm=column_adm,
                                           statistic=data_settings['algorithm']['flags']['statistic'])
        logging.info('Compute statistics by NUTS ... DONE')
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # we save the result
        logging.info(' --> Save map for ' + str(time_date) + ' ... ')
        path_csv = data_settings['data']['outcome']['path_csv']
        tags = {'outcome_sub_path_time': time_date,
                'outcome_datetime': time_date}
        path_csv = fill_tags2string(path_csv, data_settings['algorithm']['template'], tags)
        os.makedirs(os.path.dirname(path_csv), exist_ok=True)
        stats_by_nuts.to_csv(path_csv, index=False)
        logging.info(' --> Save map for ' + str(time_date) + ' ... DONE')
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


# -------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Call script from external library
if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------
