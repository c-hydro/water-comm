"""
water_comm - tool to compute and plot SWE anomaly
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
from datetime import timedelta
from datetime import date
import numpy as np

from src.json_utilities.water_comm_utilities_json import read_file_json
from src.time_utilities.water_comm_utilities_time import set_time
from src.geo_utilities.water_comm_utilities_geo import read_file_raster
from src.generic_utilties.water_comm_generic_utilities import fill_tags2string
from src.figure_utilities.water_comm_utilities_SWE_plot import SWE_anomaly_plot
from src.geo_utilities.water_comm_utilities_read_SWE import load_SWE_map_or_csv
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Algorithm information
alg_project = 'water_comm'
alg_name = 'SWE ANOMALY'
alg_version = '1.0.0'
alg_release = '2025-03-31'
alg_type = 'SWEANOMALY'
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
    # Load areacell
    logging.info(' --> Load target areacell ... ')
    da_areacell, wide_areacell, high_areacell, proj_areacell, transform_areacell, \
        bounding_box_areacell, no_data_areacell, crs_areacell, lons_areacell, lats_areacell = \
        read_file_raster(data_settings['data']['input_grid']['areacell'],
                         coord_name_x='lon', coord_name_y='lat',
                         dim_name_x='lon', dim_name_y='lat')
    logging.info(' --> Load target areacell ... DONE')
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Load csv with keys
    logging.info(' --> Load target csv ... ')
    keys = pd.read_csv(data_settings['data']['input_grid']['csv_grid_keys'])
    logging.info(' --> Load target csv ... DONE')
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Iterate over time steps, load SWE map, compute SWE for each region
    SWE_history = None
    SWE_relevant = None
    for time_i, time_date in enumerate(time_range):

        logging.info('Generating plot for time step ' + str(time_date))
        path_SWE = data_settings['data']['input_SWE_list'][:]
        tags_template = data_settings['algorithm']['template']
        csv_SWE_name = data_settings['data']['outcome']['csv_SWE_name']
        recompute_SWE = data_settings['algorithm']['flags']['recompute_SWE']
        days_enforce_recomputation = data_settings['algorithm']['flags']['days_enforce_recomputation']

        # -------------------------------------------------------------------------------------
        # Management of historical time period
        yr_start_history = data_settings['algorithm']['flags']['first_water_year_climatology']
        yr_start_history = int(yr_start_history)
        yr_end_history = data_settings['algorithm']['flags']['last_water_year_climatology']
        if yr_end_history == "now":
            if start_time_pd.month < 9:
                yr_end_history = start_time_pd.year - 1
            else:
                yr_end_history = start_time_pd.year
        else:
            yr_end_history = int(yr_end_history)
        period_history = pd.date_range(start=datetime(yr_start_history - 1, 9, 1,
                                                  time_date.hour, time_date.minute,
                                                  time_date.second),
                                       end= datetime(yr_end_history, 8, 31, time_date.hour, time_date.minute,
                                                  time_date.second), freq='D')
        logging.info(' --> Historical period: ' + str(period_history[0]) + ' - ' + str(period_history[-1]))
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # SWE extraction for historical period
        if SWE_history is not None:
            logging.info('No need to load again SWE map for historical period ... ')
        else:
            logging.info(' --> Load SWE map for historical period ... ')
            SWE_history = load_SWE_map_or_csv(period_history, time_date,
                                          keys, path_SWE.copy(), tags_template.copy(),
                                          csv_SWE_name,
                        recompute_SWE, da_domain_target, da_areacell,
                        days_enforce_recomputation)
            logging.info(' --> Load SWE map for historical period ... DONE')
        # # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # Management of current time period
        if time_date.month < 9:
            current_year_start = datetime(time_date.year - 1, 9, 1,
                                          time_date.hour, time_date.minute,
                                          time_date.second)
        else:
            current_year_start = datetime(time_date.year, 9, 1,
                                          time_date.hour, time_date.minute,
                                          time_date.second)
        period_current = pd.date_range(start=current_year_start,
                                       end= time_date, freq='D')
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # SWE extraction for current period
        logging.info(' --> Load SWE map for current period ... ')
        SWE_current = load_SWE_map_or_csv(period_current, time_date,
                                          keys, path_SWE.copy(), tags_template.copy(),
                                          csv_SWE_name,
                                          recompute_SWE, da_domain_target, da_areacell,
                                          days_enforce_recomputation)

        logging.info(' --> Load SWE map for current period ... DONE')
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # Management of relevant years to plot
        relevant_years_to_plot = data_settings['algorithm']['flags']['relevant_years_to_plot']
        if SWE_relevant is not None:
            logging.info('No need to load again SWE map for relevant years ... ')
        else:
            logging.info(' --> Load SWE map for relevant years ... ')
            SWE_relevant = {}
            for i_year, year in enumerate(relevant_years_to_plot):
                 if year == "last":
                     if time_date.month < 9:
                         period_relevant = pd.date_range(start=datetime(time_date.year - 2, 9, 1,
                                                                        time_date.hour, time_date.minute,
                                                                        time_date.second),
                                                         end= datetime(time_date.year - 1, 8, 31,
                                                                       time_date.hour, time_date.minute,
                                                                         time_date.second), freq='D')
                     else:
                         period_relevant = pd.date_range(start=datetime(time_date.year - 1, 9, 1,
                                                                        time_date.hour, time_date.minute,
                                                                        time_date.second),
                                                         end=datetime(time_date.year, 8, 31,
                                                                      time_date.hour, time_date.minute,
                                                                        time_date.second), freq='D')
                     last_year_string = str(period_relevant[-1].year)
                 else:
                     period_relevant = pd.date_range(start=datetime(int(year) - 1, 9, 1,
                                                                    time_date.hour, time_date.minute,
                                                                    time_date.second),
                                                     end= datetime(int(year), 8, 31,
                                                                   time_date.hour, time_date.minute,
                                                                   time_date.second), freq='D')

                 SWE_relevant[year] = load_SWE_map_or_csv(period_relevant, time_date,
                                                          keys, path_SWE, tags_template, csv_SWE_name,
                                                          recompute_SWE, da_domain_target, da_areacell,
                                                          days_enforce_recomputation)
            logging.info(' --> Load SWE map for relevant years ... DONE')
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # now we work on the final plot
        df_SWE_anomaly = pd.DataFrame(index=keys['Name'], columns=['anomaly'])
        for i_region, region in keys.iterrows():
            df_SWE_this_region_history = SWE_history[region['Name']]
            df_SWE_this_region_current = SWE_current[region['Name']]
            df_SWE_this_region_relevant = {}
            for i_year, year in enumerate(relevant_years_to_plot):
                df_SWE_this_region_relevant[year] = SWE_relevant[year][region['Name']]
            path_out = data_settings['data']['outcome']['figure']
            tags = {'outcome_sub_path_time': time_date,
                    'outcome_datetime': time_date,
                    'region': region['Name']}
            path_out = fill_tags2string(path_out, data_settings['algorithm']['template'], tags)
            anomaly_SWE_this_basin = (
                SWE_anomaly_plot(df_SWE_this_region_history.copy(), df_SWE_this_region_current.copy(),
                                 df_SWE_this_region_relevant.copy(), yr_start_history, yr_end_history,
                                 last_year_string,
                                 region['Name'], time_date,data_settings['data']['outcome']['logo'], path_out))
            df_SWE_anomaly.loc[region['Name'], 'anomaly'] = anomaly_SWE_this_basin
            logging.info(' --> Plotting anomaly for region ' + region['Name'] + ' ... DONE')

        path_SWE_deficit_all = os.path.split(data_settings['data']['outcome']['figure'])[0]
        path_SWE_deficit_all = fill_tags2string(path_SWE_deficit_all, data_settings['algorithm']['template'], tags)
        df_SWE_anomaly.to_csv(os.path.join(path_SWE_deficit_all, 'SWE_deficit_all.csv'))
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
