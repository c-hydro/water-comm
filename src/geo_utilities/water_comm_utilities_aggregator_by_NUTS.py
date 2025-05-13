import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from typing import Optional

from dam import as_DAM_process

@as_DAM_process(input_type = 'xarray', output_type = 'csv')
def aggregator_by_NUTS(input: xr.DataArray,
                      shapes: gpd.GeoDataFrame,
                      statistic: str = 'median',
                      all_touched: bool = False,
                      column_out: Optional[str] = None,
                      column_adm: str = None,
                      thr_quantile: Optional[float] = 0.2,
                      ) -> pd.DataFrame:
    """
    Summarise a raster by a shapefile.
    """

    # I need to check if the column_adm is None
    if column_adm is None:
        raise ValueError('The column_adm parameter must be provided.')

    # get no_data value
    nodata_value = input.attrs.get('_FillValue', np.nan)

    # Loop over each geometry in the GeoDataFrame
    for geom in shapes.geometry:
        # Mask the raster with the current geometry
        out_image = input.rio.clip([geom],
                                   all_touched=all_touched)

        # we only care about the data, not the shape
        out_data = out_image.values.flatten()

        # check if all values are nodata
        if np.all(np.isclose(out_data, nodata_value, equal_nan=True)):
            # if statistic is mode set to 0 else set to nan
            if statistic == 'mode':
                stat = 0
            else:
                stat = np.nan
        else:
            # remove the nodata values
            data = out_data[~np.isclose(out_data, nodata_value, equal_nan=True)]

            if statistic == 'mean':
                stat = np.mean(data)
            elif statistic == 'median':
                stat = np.median(data)
            elif statistic == 'mode':
                # check if data values are integers
                if np.all(np.equal(np.mod(data, 1), 0)):
                    stat = np.bincount(data.astype(int)).argmax()
                else:
                    raise ValueError('The "mode" statistic can only be calculated for integer data.')
            elif statistic == 'sum':
                stat = np.sum(data)
            elif statistic == 'quantile':
                stat = np.quantile(data, thr_quantile)
            else:
                raise ValueError('The statistic must be either "mean", "median", "mode", "sum" or "quantile".')

        if column_out is None:
            # if both threshold quantile and value are not provided
            column_out = f'{statistic}'

        shapes.loc[shapes.geometry == geom, column_out] = stat

    # Convert gdp.GeoDataFrame to pd.DataFrame keeping only the columns of interest
    shapes = shapes[[column_adm, column_out]]

    return shapes