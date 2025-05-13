# -------------------------------------------------------------------------------------
# Library
import logging
import numpy as np
import rasterio
import rasterio.crs
from rasterio.enums import Resampling
import os
import pandas as pd
from datetime import date
import json
import xarray as xr
from copy import deepcopy
from datetime import datetime

logging.getLogger('rasterio').setLevel(logging.WARNING)

# -------------------------------------------------------------------------------------
# Method to set time run
def set_time(time_run_args=None, time_run_file=None, time_format='%Y-%m-%d %H:$M',
             time_run_file_start=None, time_run_file_end=None,
             time_period=1, time_frequency='H', time_rounding='H', time_reverse=True):

    logging.info(' ----> Set time period ... ')
    if (time_run_file_start is None) and (time_run_file_end is None):

        logging.info(' -----> Time info defined by "time_run" argument ... ')

        if time_run_args is not None:
            time_run = time_run_args
            logging.info(' ------> Time ' + time_run + ' set by argument')
        elif (time_run_args is None) and (time_run_file is not None):
            time_run = time_run_file
            logging.info(' ------> Time ' + time_run + ' set by user')
        elif (time_run_args is None) and (time_run_file is None):
            time_now = date.today()
            time_run = time_now.strftime(time_format)
            logging.info(' ------> Time ' + time_run + ' set by system')
        else:
            logging.info(' ----> Set time period ... FAILED')
            logging.error(' ===> Argument "time_run" is not correctly set')
            raise IOError('Time type or format is wrong')

        time_tmp = pd.Timestamp(time_run)
        time_run = time_tmp.floor(time_rounding)

        if time_period > 0:
            time_range = pd.date_range(end=time_run, periods=time_period, freq=time_frequency)
        else:
            logging.warning(' ===> TimePeriod must be greater then 0. TimePeriod is set automatically to 1')
            time_range = pd.DatetimeIndex([time_now], freq=time_frequency)

        logging.info(' -----> Time info defined by "time_run" argument ... DONE')

    elif (time_run_file_start is not None) and (time_run_file_end is not None):

        logging.info(' -----> Time info defined by "time_start" and "time_end" arguments ... ')

        time_run_file_start = pd.Timestamp(time_run_file_start)
        time_run_file_start = time_run_file_start.floor(time_rounding)
        time_run_file_end = pd.Timestamp(time_run_file_end)
        time_run_file_end = time_run_file_end.floor(time_rounding)

        time_now = date.today()
        time_run = time_now.strftime(time_format)
        time_run = pd.Timestamp(time_run)
        time_run = time_run.floor(time_rounding)
        time_range = pd.date_range(start=time_run_file_start, end=time_run_file_end, freq=time_frequency)

        logging.info(' -----> Time info defined by "time_start" and "time_end" arguments ... DONE')

    else:
        logging.info(' ----> Set time period ... FAILED')
        logging.error(' ===> Arguments "time_start" and/or "time_end" is/are not correctly set')
        raise IOError('Time type or format is wrong')

    if time_reverse:
        time_range = time_range[::-1]

    time_chunks = set_chunks(time_range)

    logging.info(' ----> Set time period ... DONE')

    return time_run, time_range, time_chunks

# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to set chunks
def set_chunks(time_range, time_period='D'):

    time_groups = time_range.to_period(time_period)
    time_chunks = time_range.groupby(time_groups)

    return time_chunks
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
# Method to generate list of times based on wayer years
def def_dates_water_year(wy_start, wy_end, date, start_time_pd):

    # we first compute the actual calendar_year_start, which depends on the month of time_date
    if date.month < 9:
        calendar_year_start_history = wy_start
    else:
        calendar_year_start_history = wy_start - 1

    # as for the final calendar_year_end, we look to the setting of end_history
    if wy_end == 'now':
        calendar_year_end_history = start_time_pd.year
    else:
        if date.month < 9:
            calendar_year_end_history = int(wy_end)
        else:
            calendar_year_end_history = int(wy_end) - 1

    # now we compute the datetimeindex of the various dates to process
    period_all_history = []
    for year in range(calendar_year_start_history, calendar_year_end_history + 1):
        date_candidate = datetime(year, date.month, date.day, date.hour, date.minute)
        period_all_history.append(date_candidate)
    period_all_history_index = pd.DatetimeIndex(period_all_history)  # this is the final datetime index to process

    if wy_end == 'now':
        period_all_history_index = period_all_history_index[period_all_history_index <= start_time_pd]

    # and also compute a list of the respective water years, which will be convenient later
    if date.month < 9:
        list_water_years_history = period_all_history_index.year
    else:
        list_water_years_history = period_all_history_index.year + 1  # this is the list of water years to process

    logging.info(' --> List of water years to process as HISTORY: ' + str(list_water_years_history))

    return period_all_history_index, list_water_years_history











# -----------------------------------------------------------------------------------



