"""
water_comm - tool to compute SWE anomaly map
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
import numpy as np
import xarray as xr
from dateutil.relativedelta import relativedelta
np.set_printoptions(legacy='1.25')

from dam.utils.io_geotiff import write_geotiff

from src.json_utilities.water_comm_utilities_json import read_file_json
from src.time_utilities.water_comm_utilities_time import set_time, def_dates_water_year
from src.geo_utilities.water_comm_utilities_geo import read_file_raster
from src.generic_utilties.water_comm_generic_utilities import fill_tags2string
from src.geo_utilities.water_comm_utilities_read_SWE import load_SWE_map
from src.figure_utilities.water_comm_utilities_SWE_plot import histogram_SWE_deficit
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Algorithm information
alg_project = 'water_comm'
alg_name = 'SWE ANOMALY MAP'
alg_version = '1.0.0'
alg_release = '2025-04-10'
alg_type = 'SWEANOMALYMAP'
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
    start_time_pd = pd.to_datetime(start_time, unit='s')

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
    # Do the work
    for time_i, time_date in enumerate(time_range):

        # -------------------------------------------------------------------------------------
        # we immediately handle leap years by changing Feb 29 to Feb 28 just in case
        if time_date.month == 2 and time_date.day == 29:
            time_date = datetime(time_date.year, 2, 28, time_date.hour, time_date.minute)
            logging.warning('Leap year detected. Feb 29 changed to Feb 28')
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # Management of historical time period (this is a bit tricky and depends on whether history is fixed or not)
        start_history = data_settings['algorithm']['flags']['first_water_year_climatology']
        start_history = int(start_history)
        end_history = data_settings['algorithm']['flags']['last_water_year_climatology']
        period_history_index, list_water_years_history = def_dates_water_year(
            start_history, end_history, time_date, start_time_pd)
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # we also prepare the datetimeindex for relevant years
        relevant_years = data_settings['algorithm']['flags']['relevant_years']
        period_relevant_years = []
        for i_year, year in enumerate(relevant_years):
            if year == 'last':
                period_relevant_years.append(time_date - relativedelta(years=1))
            elif year == 'climatology':
                period_relevant_years.append(None)
            else:
                if time_date.month < 9:
                    period_relevant_years.append(datetime(int(year), time_date.month,
                                                        time_date.day, time_date.hour, time_date.minute))
                else:
                    period_relevant_years.append(datetime(int(year) - 1, time_date.month,
                                                        time_date.day, time_date.hour, time_date.minute))
        period_relevant_years_index = pd.DatetimeIndex(period_relevant_years)
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # We compute current water year
        if time_date.month < 9:
            this_water_year = time_date.year
        else:
            this_water_year = time_date.year + 1
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        #we load SWE maps (history)
        tags_template = data_settings['algorithm']['template']
        path_SWE = data_settings['data']['input_SWE_list'][:]  # this is in principle a list of files
        SWE_all_years_history = load_SWE_map(period_history_index, high_domain_target, wide_domain_target,
                                             path_SWE.copy(), tags_template, no_data_domain_target, da_domain_target)
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # we load SWE maps (this year)
        logging.info(' --> Load SWE map for this year ... ')
        path_SWE = data_settings['data']['input_SWE_list'][:]
        SWE_this_year = load_SWE_map(time_date, high_domain_target, wide_domain_target,
                                             path_SWE.copy(), tags_template, no_data_domain_target, da_domain_target)
        SWE_this_year = np.squeeze(SWE_this_year)
        logging.info(' --> Load SWE map for this year ... DONE')
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # we load SWE maps (relevant years)
        logging.info(' --> Load SWE map for relevant years ... ')
        SWE_relevant = {}
        for i_year, year in enumerate(relevant_years):
            if year == 'climatology':
                mask = np.array(list_water_years_history) != this_water_year
                SWE_filtered = SWE_all_years_history[mask, :, :]
                SWE_relevant[year] = np.nanmean(SWE_filtered, axis=0)
                logging.info(' --> SWE climatology computed')
            else:
                time_relevant_year = period_relevant_years_index[i_year]
                SWE_relevant[year] = np.squeeze(load_SWE_map(time_relevant_year, high_domain_target, wide_domain_target,
                                             path_SWE.copy(), tags_template, no_data_domain_target, da_domain_target))
        logging.info(' --> Load SWE map for relevant years ... DONE')
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # now we compute anomalies by looping on this dictionary & save maps & some summary plots
        threshold_SWE = data_settings['algorithm']['flags']['threshold_SWE']
        relevant_years_tags = data_settings['algorithm']['flags']['relevant_years_tags']
        for i_year, year in enumerate(relevant_years):

            SWE_anomaly_tmp = (SWE_this_year - SWE_relevant[year]) / SWE_relevant[year] * 100
            SWE_anomaly_tmp[SWE_relevant[year] < threshold_SWE] = np.nan
            logging.info(' --> SWE anomaly compared to year ' + str(year) + ' computed')
            logging.info('Mean anomaly with compared to year ' + str(year) + ' is ' + str(np.nanmean(SWE_anomaly_tmp)))
            logging.info('Max anomaly with compared to year ' + str(year) + ' is ' + str(np.nanmax(SWE_anomaly_tmp)))
            logging.info('Q1 anomaly with compared to year ' + str(year) + ' is ' + str(np.nanquantile(SWE_anomaly_tmp, 0.25)))
            logging.info('Q2 anomaly with compared to year ' + str(year) + ' is ' + str(np.nanquantile(SWE_anomaly_tmp, 0.5)))
            logging.info('Q3 anomaly with compared to year ' + str(year) + ' is ' + str(np.nanquantile(SWE_anomaly_tmp, 0.75)))

            # save map
            output_dir = data_settings['data']['outcome']['path']
            tags = {'outcome_sub_path_time': time_date,
                    'outcome_datetime': time_date,
                    'tag_benchmark': relevant_years_tags[i_year]}
            output_dir = fill_tags2string(output_dir, data_settings['algorithm']['template'], tags)
            layer_out = SWE_anomaly_tmp.astype(np.float32)
            layer_out[np.isnan(layer_out)] = data_settings['data']['outcome']['no_data_value']
            layer_out_xr = xr.DataArray(layer_out, dims=['y', 'x'], coords={'y': da_domain_target.lat.values, 'x': da_domain_target.lon.values})
            layer_out_xr = layer_out_xr.rio.write_crs("EPSG:4326")
            layer_out_xr = layer_out_xr.rio.write_nodata(data_settings['data']['outcome']['no_data_value'])
            write_geotiff(layer_out_xr, output_dir)
            logging.info(" --> SWE anomaly map saved to " + str(output_dir))

            # creat histogram and save
            histogram_SWE_deficit(SWE_anomaly_tmp, output_dir, time_date, year)

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
