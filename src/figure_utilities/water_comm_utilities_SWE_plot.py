# -------------------------------------------------------------------------------------
# Library
import logging
import numpy as np
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logging.getLogger('rasterio').setLevel(logging.WARNING)

# -------------------------------------------------------------------------------------
# Method to read file json
def SWE_anomaly_plot(df_history, df_current, df_data_relevant,
                     yr_start_history, yr_end_history, last_yr_string, region,
                     date_now, logo, path_out,
                     unit_yaxis='$10^9$ m$^3$', scale_yaxis=1e-9):

    # -------------------------------------------------------------------------------------
    # let' start by computing quartiles of df_history for each day of the year
    df_history.index = pd.to_datetime(df_history.index)
    df_history = df_history.to_frame()
    df_history = df_history[~((df_history.index.month == 2) & (df_history.index.day == 29))] # we remove Feb 29
    df_history['dayofyear'] = df_history.index.day_of_year
    df_history.loc[(df_history.index.is_leap_year) & (df_history.index.month > 2), 'dayofyear'] -= 1
    df_history = df_history.apply(pd.to_numeric, errors='coerce')
    df_history = df_history.groupby('dayofyear').quantile([0.25, 0.5, 0.75]).unstack(level=-1)
    df_history.columns = ['Q1', 'Q2', 'Q3']

    # now we rearrange this dataframe to have september 1 as day 1
    df_history = df_history.reset_index()  # Reset index to manipulate dayofyear
    df_history['dayofyear'] = (df_history['dayofyear'] - 244) % 365  # Shift and wrap around
    df_history['dayofyear'] = df_history['dayofyear'].replace(0, 365)  # Handle day 0 as day 365
    df_history = df_history.sort_values('dayofyear').set_index('dayofyear')  # Sort and set as index

    # smooth data using a 15-day rolling mean, centered on the day
    df_history = df_history.rolling(15, center=True, min_periods=1).mean()
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # now we define the dataframe for the current year
    df_current = df_current[~((df_current.index.month == 2) & (df_current.index.day == 29))]
    df_current = df_current.to_frame()
    df_current = df_current.reset_index(drop=True)
    df_current['index'] = df_history.index[0:np.size(df_current)]
    df_current = df_current.set_index('index')
    df_current = df_current.rolling(15, center=True, min_periods=1).mean()
    if date_now.month < 9:
        wy_now = date_now.year
    else:
        wy_now = date_now.year + 1
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # now we also prepare the df with the relevant yrs
    for year, data in df_data_relevant.items():
        data = data[~((data.index.month == 2) & (data.index.day == 29))]
        data = data.to_frame()
        data = data.reset_index(drop=True)
        data['index'] = df_history.index
        data = data.set_index('index')
        data = data.rolling(15, center=True, min_periods=1).mean()
        df_data_relevant[year] = data
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # now we plot the data
    fig, ax = plt.subplots(figsize=(14, 6.5))

    # history
    ax.fill_between(df_history.index, df_history['Q1']*scale_yaxis, df_history['Q3']*scale_yaxis, color='gray', alpha=0.5,
                    label= str(yr_start_history) + '-'
                           + str(yr_end_history) + ' Q1-Q3')
    ax.plot(df_history.index, df_history['Q2']*scale_yaxis,
            color='black', label= str(yr_start_history) + '-'
            + str(yr_end_history) + ' Q2', linewidth=2)

    # other years
    # Adjust grayscale range to avoid extremes
    grayscale_start = 0.5  # Darker grey
    grayscale_end = 0.8  # Lighter grey
    grayscale_values = np.linspace(grayscale_start, grayscale_end, len(df_data_relevant))
    for i, (year, data) in enumerate(df_data_relevant.items()):
        if year == "last":
            line_style = '--'
            label_year = last_yr_string
        else:
            line_style = '-'
            label_year = year
        if df_data_relevant.__len__() > 1:
            color_value = plt.cm.Greys(grayscale_values[i])
        else:
            color_value = 'gray'
        ax.plot(df_history.index, data*scale_yaxis,
                color=color_value,
                label=label_year, linewidth=2, linestyle=line_style)

    # present
    ax.plot(df_current.index, df_current*scale_yaxis, color='red', label=wy_now, linewidth=2.5)
    ax.plot(df_current.index[-1], df_current.iloc[-1]*scale_yaxis, 'r*', markersize=15, markeredgecolor='black',
            markeredgewidth=1.5, label=date_now.strftime('%Y/%b/%d'))

    #legend
    ax.legend(loc='upper right', fontsize=13, frameon=True, facecolor='white', edgecolor='black')

    # compute anomaly and place title
    anomaly = (df_current.iloc[-1] - df_history.loc[df_current.index[-1], 'Q2']) / df_history.loc[df_current.index[-1], 'Q2'] * 100
    anomaly = anomaly.values[0]
    if anomaly > 0:
         ax.set_title(region + ': snow-water-equivalent surplus: +' + str(round(anomaly, 2)) + '%', fontsize=15)
    else:
        ax.set_title(region + ': snow-water-equivalent deficit: ' + str(round(anomaly, 2)) + '%', fontsize=15)

    #y axis
    ax.set_ylabel('SWE, ' + unit_yaxis, fontsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_ylim(bottom=0)

    #x axis
    start_date = pd.to_datetime('2021-09-01') #fictitious start date, non leap year
    dates = pd.date_range(start_date, periods=365)
    first_days_of_months = dates[dates.is_month_start]
    date_labels = first_days_of_months.strftime('%b %d')
    ax.set_xticks(np.sort(first_days_of_months.dayofyear))
    ax.set_xticklabels(date_labels, ha='right', fontsize=15, rotation=45)
    ax.set_xlim(1, 365)

    #logo
    if logo is not None:
        logo = plt.imread(logo)
        ax_logo = ax.inset_axes([0, 0.75, 0.2, 0.2],
                                transform=ax.transAxes, zorder=10)
        ax_logo.imshow(logo)
        ax_logo.axis('off')

    ax.annotate(
        'Q1, Q2, and Q3 are the first, second, and third quartiles of daily SWE during the historical period. Years are water years, starting on September 1. \n '
        '(c) CIMA Research Foundation, regular updates are posted during winter here: https://www.cimafoundation.org/en/italy-snow-updates/ \n '
        'Remember: today\'s snow is tomorrow\'s water! \n ',
        xy=(1.0, -0.27),  # Position: x=1.0 (far right), y=-0.2 (below x-axis)
        xycoords='axes fraction',  # Coordinates relative to the axes
        ha='right',  # Horizontal alignment
        fontsize=9  # Font size
    )

    #save fig
    folder_out = os.path.dirname(path_out)
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    plt.savefig(path_out, bbox_inches='tight', dpi=300)

    #create output dataframe and then save as csv
    df_out = pd.DataFrame(index=df_history.index)
    df_out['History, Q1'] = df_history['Q1']*scale_yaxis
    df_out['History, Q2'] = df_history['Q2']*scale_yaxis
    df_out['History, Q3'] = df_history['Q3']*scale_yaxis
    df_out['Current year'] = df_current*scale_yaxis
    for i, (year, data) in enumerate(df_data_relevant.items()):
        df_out[year] = df_data_relevant[year]*scale_yaxis
    df_out['Anomaly'] = np.nan
    df_out['Anomaly'][0:np.size(df_current)] = ((np.squeeze(df_current.values) - df_history['Q2'].values[0:np.size(df_current)])/
                         df_history['Q2'].values[0:np.size(df_current)] * 100)
    df_out = df_out.round(3)
    path_out_csv = os.path.splitext(path_out)[0] + '.csv'
    df_out.to_csv(path_out_csv, index=True)

    return anomaly
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def histogram_SWE_deficit(SWE_anomaly, output_dir, date,
                          year_reference, SWE_anomaly_max=500, special_value=-100,
                          special_value_label='Values equal to -100%',
                          non_special_value_label='Values above -100%'):

    # Filter data
    SWE_for_hist = SWE_anomaly[(~np.isnan(SWE_anomaly)) & (SWE_anomaly < SWE_anomaly_max)]
    special_value_count = np.sum(SWE_for_hist == special_value)  # Count values equal to -100
    non_special_data = SWE_for_hist[SWE_for_hist != special_value]  # Exclude -100 for histogram

    # Create figure and axes
    fig, (ax_high, ax_low) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 3]})
    fig.subplots_adjust(hspace=0.05)  # Adjust space between axes

    if special_value_count > 0:
        special_density = special_value_count / len(SWE_for_hist)
        ax_high.bar(-100, special_density, width=5, color='red', alpha=0.5, label=special_value_label)

    sns.histplot(non_special_data, kde=True, color='blue', alpha=0.3, ax=ax_low,
                 stat='density', label=non_special_value_label)
    if special_value_count > 0:
        ax_low_ylim = ax_low.get_ylim()
        ax_low.bar(-100, special_density, width=5, color='red', alpha=0.5)

    ax_low.set_ylim(0, ax_low_ylim[1])  # Focus on non-special density range
    ax_high.set_ylim(ax_low_ylim[1], special_density * 1.2)  # Add space for special value bar

    ax_high.spines.bottom.set_visible(False)
    ax_low.spines.top.set_visible(False)
    ax_high.xaxis.tick_top()
    ax_high.tick_params(labeltop=False)  # don't put tick labels at the top
    ax_low.xaxis.tick_bottom()

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax_high.plot([0, 1], [0, 0], transform=ax_high.transAxes, **kwargs)
    ax_low.plot([0, 1], [1, 1], transform=ax_low.transAxes, **kwargs)

    mean_anomaly = np.nanmean(SWE_for_hist)

    ax_low.set_xlabel('SWE Anomaly, %')
    ax_low.set_ylabel('Density, -')
    ax_low.legend(loc='upper right')
    fig.suptitle('SWE Anomaly for ' + str(date) + ' compared to ' +
                 str(year_reference)+ ': ' + str(np.round(mean_anomaly, 2)) + '%'),

    # Save the figure
    output_dir_hist = output_dir.replace('.tif', '.png')
    fig.savefig(output_dir_hist, dpi=300)
    plt.close(fig)


# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def anomaly_elevation_plot(df, path, region, date):

    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.fill_between((df.elevation_band_down + df.elevation_band_up) / 2,
                    df['percentile_25'], df['percentile_75'], color='gray',
                    alpha=0.5,
                    label='Q1-Q3 anomaly')
    ax.plot((df.elevation_band_down + df.elevation_band_up) / 2,
            df['percentile_50'], color='black', label='Q2 anomaly', linewidth=2,
            linestyle='-', marker='o', markerfacecolor='black')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)

    ax.set_ylim(bottom=-100)
    ax.set_ylim(top=150)
    ax.set_xlim(left=np.nanmin((df.elevation_band_down + df.elevation_band_up) / 2),
                right=np.nanmax((df.elevation_band_down + df.elevation_band_up) / 2))
    ax.set_xlim(right=4000)
    ax.set_ylabel('Anomaly, %', fontsize=20)
    ax.set_xlabel('Elevation, m', fontsize=20)
    ax.set_title(region + ': SWE Anomaly by Elevation, date:' + date.strftime('%Y/%b/%d'), fontsize=20)
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='x', labelsize=15)
    ax.legend(loc='upper right', fontsize=13, frameon=True, facecolor='white', edgecolor='black')

    # Save the figure
    fig.savefig(path, dpi=300)
    plt.close(fig)

# -------------------------------------------------------------------------------------