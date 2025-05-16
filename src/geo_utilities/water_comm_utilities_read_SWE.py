# -------------------------------------------------------------------------------------
# Library
import logging
import numpy as np
import os
import pandas as pd
from datetime import timedelta

from src.generic_utilties.water_comm_generic_utilities import fill_tags2string
from src.geo_utilities.water_comm_utilities_geo import read_file_raster

logging.getLogger('rasterio').setLevel(logging.WARNING)

# -------------------------------------------------------------------------------------
def load_SWE_map(period, high_domain_target, wide_domain_target, path_SWE,
                 tags_template, no_data_domain_target, da_domain_target,
                 negative_values_to_zero=True):

    SWE_all = np.zeros((np.size(period), high_domain_target, wide_domain_target))*np.nan
    list_periods = np.arange(0, np.size(period))

    for i, list_value in enumerate(list_periods):

        SWE_exists = list(range(len(path_SWE)))
        path_SWE_tmp = path_SWE.copy()
        for i_SWE, path_SWE_list in enumerate(path_SWE):

            if np.size(list_periods) > 1:
                tags = {'source_gridded_sub_path_time': period[i],
                    'source_gridded_datetime': period[i]}
            else:
                tags = {'source_gridded_sub_path_time': period,
                        'source_gridded_datetime': period}

            path_SWE_tmp[i_SWE] = fill_tags2string(path_SWE_list, tags_template, tags)
            SWE_exists[i_SWE] = os.path.exists(path_SWE_tmp[i_SWE])

        selected_path_SWE = next((path for exists, path in zip(SWE_exists, path_SWE_tmp) if exists), None)
        logging.info('The candidate SWE path is ' + str(selected_path_SWE))

        if selected_path_SWE is None:
            if np.size(list_periods) > 1:
                logging.warning('No SWE map found for ' + str(period[i]))
            else:
                logging.warning('No SWE map found for this day')
            SWE_all[i, :, :] = np.zeros((high_domain_target, wide_domain_target)) * np.nan
            logging.warning(' --> Filled with NaN')
        else:
            SWE_tmp = read_file_raster(selected_path_SWE,
                                                  coord_name_x='lon', coord_name_y='lat',
                                                  dim_name_x='lon', dim_name_y='lat')
            SWE_tmp = SWE_tmp[0].values
            if negative_values_to_zero:
                SWE_tmp[SWE_tmp < 0] = 0
            SWE_tmp[da_domain_target == no_data_domain_target] = np.nan
            SWE_tmp[np.isnan(da_domain_target)] = np.nan
            SWE_all[i, :, :] = SWE_tmp

    return SWE_all

# -------------------------------------------------------------------------------------
def load_SWE_map_or_csv(period, time_date, keys, path_SWE, tags_template, csv_SWE_name,
                        recompute_SWE, da_domain_target, da_areacell,
                        days_enforce_recomputation = 15):

    SWE_this_day = None
    df_SWE = pd.DataFrame(index=period, columns=keys['Name'])

    for i_period, period_date in enumerate(period):

        # build SWE paths
        path_SWE_tmp = path_SWE[:].copy()  # this is in principle a list of files
        SWE_exists = list(range(len(path_SWE)))
        for i_SWE, path_SWE_list in enumerate(path_SWE):
            tags = {'source_gridded_sub_path_time': period_date,
                    'source_gridded_datetime': period_date}
            path_SWE_tmp[i_SWE] = fill_tags2string(path_SWE_list, tags_template, tags)
            SWE_exists[i_SWE] = os.path.exists(path_SWE_tmp[i_SWE])
        selected_path_SWE = next((path for exists, path in zip(SWE_exists, path_SWE_tmp) if exists), None)
        logging.info('The candidate SWE path is ' + str(selected_path_SWE))

        # now loop on regions and decide what to do and where to get data from
        for i_region, region in keys.iterrows():

            logging.info('Computing SWE for region ' + region['Name'])

            # first we generate the candidate csv paths
            csv_path = list(range(len(path_SWE)))
            csv_exists = list(range(len(path_SWE)))
            for i_SWE, path_SWE_list in enumerate(path_SWE):
                csv_path[i_SWE] = os.path.join(os.path.dirname(path_SWE_list), csv_SWE_name)
                tags = {'source_gridded_sub_path_time': period_date,
                        'source_gridded_datetime': period_date,
                        'region': region['Name']}
                csv_path[i_SWE] = fill_tags2string(csv_path[i_SWE], tags_template, tags)
                csv_exists[i_SWE] = os.path.exists(csv_path[i_SWE])
            selected_path_csv = next((path for exists, path in zip(csv_exists, csv_path) if exists), None)
            logging.info('The candidate csv path is ' + str(selected_path_csv))

            if (selected_path_csv is None) & (selected_path_SWE is None):

                logging.warning('No tiff or csv file found for ' + region['Name'] + ' at time ' + str(period_date))

            else:

                # it's about time to decide where to get data from based on al these paths
                if ((selected_path_csv is not None) & (recompute_SWE is False)
                        & (period_date < time_date - timedelta(
                            days=days_enforce_recomputation))):

                    # which means: we have a csv file and we can use it bc we don't need to recompute SWE
                    logging.info('SWE for this region will be taken from the csv file' + str(selected_path_csv))
                    SWE_this_region_this_day = pd.read_csv(selected_path_csv, index_col=0)
                    df_SWE.loc[period_date, region['Name']] = SWE_this_region_this_day[region['Name']].values[0]

                else:
                    if SWE_this_day is None:
                        # we will load SWE map for this day and use it
                        logging.info('SWE for this region will be taken from the SWE map' + str(selected_path_SWE))
                        SWE_this_day = read_file_raster(selected_path_SWE, coord_name_x='lon', coord_name_y='lat',
                                                        dim_name_x='lon', dim_name_y='lat')[0]
                        SWE_this_day = SWE_this_day.values
                        SWE_this_day[SWE_this_day < 0] = 0

                    SWE_this_region_this_day = (
                        np.nansum(SWE_this_day[da_domain_target == region['ID']] / 1000 *
                                  da_areacell.values[da_domain_target == region['ID']]))
                    df_SWE.loc[period_date, region['Name']] = SWE_this_region_this_day

                    # let's save the csv file (making sure we are using the same location as selected_path_SWE
                    path_csv = os.path.join(os.path.dirname(selected_path_SWE),csv_SWE_name)
                    tags = {'source_gridded_sub_path_time': period_date,
                            'source_gridded_datetime': period_date,
                            'region': region['Name']}
                    path_csv = fill_tags2string(path_csv, tags_template, tags)
                    df_SWE_this_region_this_day = pd.DataFrame(index=[period_date], columns=[region['Name']])
                    df_SWE_this_region_this_day.loc[period_date, region['Name']] = SWE_this_region_this_day
                    df_SWE_this_region_this_day.to_csv(path_csv)

        SWE_this_day = None

    return df_SWE