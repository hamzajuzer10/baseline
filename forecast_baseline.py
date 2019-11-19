import pandas as pd
from multiprocessing import Process, Manager
import pandas_gbq
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import logging
import warnings

# Input global variables 
# Reformat code to accept these variables as input

# Project ID
project_id = "gum-eroski-dev"

# Define key forward baseline parameters

# Scope for the baseline (at an area level)
bl_s = "ALIMENTACION"

# Append or replace destination table (either 'append' or 'replace')
bl_table_config = 'replace'

# Forward looking period (wks)
forward_period = 16

# Seasonal period
seasonal_period = 12

# Frequency
freq='W-MON'

# Constant value to add to input data
constant = 1000

# Set batch size
batchsize = 50

# Set logger properties
logger = logging.getLogger('forecast_baseline_calculation')
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.FileHandler('forecast_baseline.log')
fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

# Function to load distinct historic baseline section data at a sku level from bigquery
def load_t0_from_bq(area, project_id):
    start_time = time.time()

    summary_sql = """
    SELECT distinct sku.section 
    FROM `gum-eroski-dev.baseline_performance.baseline` bline
    INNER JOIN `gum-eroski-dev.ETL.root_sku` sku
    ON sku.sku_root_id = bline.sku_root_id
    WHERE sku.area = "%s"   """ %(area)
    start = time.time()

    for i in tqdm(range(1), desc='Loading table...'):
        section_table = pandas_gbq.read_gbq(summary_sql, project_id=project_id)

    total_time = round((time.time() - start_time) / 60, 1)
    logger.info("Completed loading of distinct baseline sections from Bigquery in {a} mins...".format(a=total_time))

    return section_table


# Function to load baseline summary data at a sku level from bigquery
def load_t1_from_bq(section, project_id):
    start_time = time.time()

    summary_sql = """
    SELECT bline.*
    FROM `gum-eroski-dev.baseline_performance.baseline` bline
    INNER JOIN `gum-eroski-dev.ETL.root_sku` sku
    ON sku.sku_root_id = bline.sku_root_id
    WHERE sku.section = "%s"   """ %(section)
    start = time.time()

    for i in tqdm(range(1), desc='Loading table...'):
        summary_table = pandas_gbq.read_gbq(summary_sql, project_id=project_id)

    total_time = round((time.time() - start_time) / 60, 1)
    logger.info("Completed loading of baseline summary table from Bigquery in {a} mins...".format(a=total_time))

    return summary_table


# define function to process metric for one sku
def metric_baseline_sku(sku, measure, params, fit_baseline_metric, results_df_columns):
    """produce summary metrics for baseline calculation for each SKU
    :Args
        sku(str): sku_root_id
        measure(str): string identifying the measure i.e. sale_amt, sale_qty, etc
        params(list): list of params
        fit_baseline_metric(Exponential Smoothing class): Exponential smoothing class fitted on data
        results_df(Index): column index of headers in result dataframe

    :return:
        metric_series(series): series of metrics for each measure
    """
    param_list = [sku, measure]
    for p in params:
        param_list.append(fit_baseline_metric.params[p])

    # add SSE
    param_list.append(fit_baseline_metric.sse)

    # add MAE
    param_list.append(fit_baseline_metric.resid.abs().mean())

    # convergence flag
    if np.isnan(fit_baseline_metric.sse):
        param_list.append(0)
    else:
        param_list.append(1)


    # add to sku metrics dataframe
    metric_series = pd.Series(param_list, index=results_df_columns)
    return metric_series


# define function to process baseline for one sku
def forward_looking_baseline_sku(sku_pred_frame, sku_metric_frame, sku, summary_table,
                                 seasonal_period, forward_period, freq, constant = 1000):
    """produce forward looking baseline for the sku
    Args:
        sku_pred_frame(list): Multiproc manager list to store predicted baseline values
        sku_metric_frame(list_: Multiproc manager list to store predicted baseline statistics
        sku(str): sku_root_id
        summary_table(dataframe): dataframe of historic baseline values to train the model
        seasonal_period: input seasonal period (in weeks)
        forward_period: forward period to predict baseline values
        constant(int): constant value added to input to train regression model
    Returns:
        sku_pred_frame, sku_metric_frame
    """
    logger.debug(f'{sku} - being processed...')

    # get dataframe for the specific sku
    df_sku = summary_table[summary_table.sku_root_id == sku].sort_values(by=['date']).reset_index(drop=True)
    
    logger.debug(f'{sku} - extracted sku data')
    logger.debug("{sku} - extracted sku data has {rcount} rows".format(sku=sku, rcount=df_sku.shape[0]))
    
    # get the input data for the training model
    def fw_baseline_input(sku, change_flag, actual, baseline, ext):
        if change_flag == 0:
            return actual
        elif change_flag in (1, 2):
            return baseline
        elif change_flag == 3:
            return ext
        else:
            logger.error(f'{sku} - unexpected value in change_flag column!')

    df_sku['hist_baseline_sale_amt'] = df_sku.apply(lambda x: fw_baseline_input(x['sku_root_id'],
                                                                                x['change_flag'],
                                                                                x['total_sale_amt'],
                                                                                x['sale_amt_bl'],
                                                                                x['sale_amt_bl_ext']), axis=1)

    df_sku['hist_baseline_sale_qty'] = df_sku.apply(lambda x: fw_baseline_input(x['sku_root_id'],
                                                                                x['change_flag'],
                                                                                x['total_sale_qty'],
                                                                                x['sale_qty_bl'],
                                                                                x['sale_qty_bl_ext']), axis=1)

    df_sku['hist_baseline_margin_amt'] = df_sku.apply(lambda x: fw_baseline_input(x['sku_root_id'],
                                                                                  x['change_flag'],
                                                                                  x['total_margin_amt'],
                                                                                  x['margin_amt_bl'],
                                                                                  x['margin_amt_bl_ext']), axis=1)

    logger.debug(f'{sku} - completed calculation of historic baseline values...')
    

    # Split dataframe to 3 time series
    df_sku_hist_baseline_sale_amt = df_sku[['date', 'hist_baseline_sale_amt']]
    df_sku_hist_baseline_sale_amt = df_sku_hist_baseline_sale_amt.set_index('date', drop=True)

    df_sku_hist_baseline_sale_qty = df_sku[['date', 'hist_baseline_sale_qty']]
    df_sku_hist_baseline_sale_qty = df_sku_hist_baseline_sale_qty.set_index('date', drop=True)

    df_sku_hist_baseline_margin_amt = df_sku[['date', 'hist_baseline_margin_amt']]
    df_sku_hist_baseline_margin_amt = df_sku_hist_baseline_margin_amt.set_index('date', drop=True)
    
    # Convert all values to numeric
    df_sku_hist_baseline_sale_amt['hist_baseline_sale_amt'] = pd.to_numeric(df_sku_hist_baseline_sale_amt['hist_baseline_sale_amt'])
    df_sku_hist_baseline_sale_qty['hist_baseline_sale_qty'] = pd.to_numeric(df_sku_hist_baseline_sale_qty['hist_baseline_sale_qty'])
    df_sku_hist_baseline_margin_amt['hist_baseline_margin_amt'] = pd.to_numeric(df_sku_hist_baseline_margin_amt['hist_baseline_margin_amt'])
    
    logger.debug(f'{sku} - split data into timeseries...')

    # Box-cox transformations and exponential smoothing multiplicative methods require strictly positive values
    # As a workaround, for each time series, we find the smallest value and add that to the overall time series
    # We then subtract the value from the final series
    hist_baseline_sale_amt_constant = df_sku_hist_baseline_sale_amt['hist_baseline_sale_amt'].min()
    hist_baseline_sale_qty_constant = df_sku_hist_baseline_sale_qty['hist_baseline_sale_qty'].min()
    hist_baseline_margin_amt_constant = df_sku_hist_baseline_margin_amt['hist_baseline_margin_amt'].min()
    
    logger.debug(f'{sku} - computed min values for measures...')
    logger.debug("{sku} - sale_amt min value calculated to be {constant}".format(sku=sku, constant=hist_baseline_sale_amt_constant))
    logger.debug("{sku} - sale_qty min value calculated to be {constant}".format(sku=sku, constant=hist_baseline_sale_qty_constant))
    logger.debug("{sku} - margin_amt min value calculated to be {constant}".format(sku=sku, constant=hist_baseline_margin_amt_constant))

    # ensure min is at least 1000 (need to configure)
    if hist_baseline_sale_amt_constant > constant:
        hist_baseline_sale_amt_constant = 0
    elif (hist_baseline_sale_amt_constant < constant) and (hist_baseline_sale_amt_constant >= 0):
        hist_baseline_sale_amt_constant = constant
    elif hist_baseline_sale_amt_constant < 0:
        hist_baseline_sale_amt_constant = hist_baseline_sale_amt_constant*-1 + constant

    if hist_baseline_sale_qty_constant > constant:
        hist_baseline_sale_qty_constant = 0
    elif (hist_baseline_sale_qty_constant < constant) and (hist_baseline_sale_qty_constant >= 0):
        hist_baseline_sale_qty_constant = constant
    elif hist_baseline_sale_qty_constant < 0:
        hist_baseline_sale_qty_constant = hist_baseline_sale_qty_constant*-1 + constant

    if hist_baseline_margin_amt_constant > constant:
        hist_baseline_margin_amt_constant = 0
    elif (hist_baseline_margin_amt_constant < constant) and (hist_baseline_margin_amt_constant >= 0):
        hist_baseline_margin_amt_constant = constant
    elif hist_baseline_margin_amt_constant < 0:
        hist_baseline_margin_amt_constant = hist_baseline_margin_amt_constant*-1 + constant
    
    logger.debug(f'{sku} - computed constant values for measures...')
    
    # Add constants to each value in series
    df_sku_hist_baseline_sale_amt_u = df_sku_hist_baseline_sale_amt.copy()
    df_sku_hist_baseline_sale_qty_u= df_sku_hist_baseline_sale_qty.copy()
    df_sku_hist_baseline_margin_amt_u = df_sku_hist_baseline_margin_amt.copy()
    
    logger.debug(f'{sku} - copied updated measure dataframes...')
    
    logger.debug("{sku} - sale_amt_constant calculated to be {constant}".format(sku=sku, constant=hist_baseline_sale_amt_constant))
    logger.debug("{sku} - sale_qty_constant calculated to be {constant}".format(sku=sku, constant=hist_baseline_sale_qty_constant))
    logger.debug("{sku} - margin_amt_constant calculated to be {constant}".format(sku=sku, constant=hist_baseline_margin_amt_constant))

    df_sku_hist_baseline_sale_amt_u['hist_baseline_sale_amt'] += hist_baseline_sale_amt_constant
    df_sku_hist_baseline_sale_qty_u['hist_baseline_sale_qty'] += hist_baseline_sale_qty_constant
    df_sku_hist_baseline_margin_amt_u['hist_baseline_margin_amt'] += hist_baseline_margin_amt_constant

#     # Test - visualise the data
#     ax = df_sku_hist_baseline_sale_amt.plot(marker='o', color='black', figsize=(12, 8))
#     df_sku_hist_baseline_sale_amt_u.plot(marker='x', ax=ax, color='blue', legend=True)
#     df_sku_hist_baseline_sale_qty.plot(marker='o', ax=ax, color='green', legend=True)
#     df_sku_hist_baseline_sale_qty_u.plot(marker='x', ax=ax, color='red', legend=True)
#     df_sku_hist_baseline_margin_amt.plot(marker='o', ax=ax, color='yellow', legend=True)
#     df_sku_hist_baseline_margin_amt_u.plot(marker='x', ax=ax, color='cyan', legend=True)
#     plt.show()

    # Split to test and train datasets
    # TODO

    # Use Holt Winters multiplicative exponential smoothing method
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        fit_hist_baseline_sale_amt = ExponentialSmoothing(df_sku_hist_baseline_sale_amt_u, seasonal_periods=seasonal_period,
                                                          trend='add',
                                                          seasonal='add', damped=True, freq=freq).fit(use_boxcox=False,
                                                                                                      remove_bias=True)

        fit_hist_baseline_sale_qty = ExponentialSmoothing(df_sku_hist_baseline_sale_qty_u, seasonal_periods=seasonal_period,
                                                          trend='add',
                                                          seasonal='add', damped=True, freq=freq).fit(use_boxcox=False,
                                                                                                      remove_bias=True)

        fit_hist_baseline_margin_amt = ExponentialSmoothing(df_sku_hist_baseline_margin_amt_u, seasonal_periods=seasonal_period,
                                                            trend='add',
                                                            seasonal='add', damped=True, freq=freq).fit(use_boxcox=False,
                                                                                                      remove_bias=True)

    logger.debug(f'{sku} - completed fit of exponential smoothing model...')

    metrics_results_df = pd.DataFrame(columns=['sku_root_id', 'metric', 'alpha', 'beta',
                                               'phi', 'gamma', 'l_0', 'b_0', 'SSE', 'MAE', 'convergence_flag'])
    params = ['smoothing_level', 'smoothing_slope', 'damping_slope', 'smoothing_seasonal', 'initial_level',
              'initial_slope']

    metric_series_sale_amt = metric_baseline_sku(sku, 'pred_baseline_sale_amt', params,
                                                 fit_hist_baseline_sale_amt, metrics_results_df.columns)
    metric_series_sale_qty = metric_baseline_sku(sku, 'pred_baseline_sale_qty', params,
                                                 fit_hist_baseline_sale_qty, metrics_results_df.columns)
    metric_series_margin_amt = metric_baseline_sku(sku, 'pred_baseline_margin_amt', params,
                                                 fit_hist_baseline_margin_amt, metrics_results_df.columns)

    # add to sku metrics dataframe
    metrics_results_df = metrics_results_df.append(metric_series_sale_amt, ignore_index=True)
    metrics_results_df = metrics_results_df.append(metric_series_sale_qty, ignore_index=True)
    metrics_results_df = metrics_results_df.append(metric_series_margin_amt, ignore_index=True)

    # forecast x periods and view internals of the smoothing model
    # sale amt
    df_sku_pred_baseline_sale_amt = pd.DataFrame(np.c_[df_sku_hist_baseline_sale_amt,
                                                       fit_hist_baseline_sale_amt.level,
                                                       fit_hist_baseline_sale_amt.slope,
                                                       fit_hist_baseline_sale_amt.season,
                                                       fit_hist_baseline_sale_amt.fittedvalues],
                                                 columns=['y_t_sale_amt', 'l_t_sale_amt', 'b_t_sale_amt',
                                                          's_t_sale_amt', 'y_hat_t_sale_amt'],
                                                 index=df_sku_hist_baseline_sale_amt.index)

    df_sku_pred_baseline_sale_amt = df_sku_pred_baseline_sale_amt.append(
        fit_hist_baseline_sale_amt.forecast(forward_period).rename('y_hat_t_sale_amt').to_frame(), sort=True)
    #subtract constant from y_hat_t where non-null
    df_sku_pred_baseline_sale_amt['y_hat_t_sale_amt'] -= hist_baseline_sale_amt_constant


    # sale qty
    df_sku_pred_baseline_sale_qty = pd.DataFrame(np.c_[df_sku_hist_baseline_sale_qty,
                                                       fit_hist_baseline_sale_qty.level,
                                                       fit_hist_baseline_sale_qty.slope,
                                                       fit_hist_baseline_sale_qty.season,
                                                       fit_hist_baseline_sale_qty.fittedvalues],
                                                 columns=['y_t_sale_qty', 'l_t_sale_qty', 'b_t_sale_qty',
                                                          's_t_sale_qty', 'y_hat_t_sale_qty'],
                                                 index=df_sku_hist_baseline_sale_qty.index)

    df_sku_pred_baseline_sale_qty = df_sku_pred_baseline_sale_qty.append(
        fit_hist_baseline_sale_qty.forecast(forward_period).rename('y_hat_t_sale_qty').to_frame(), sort=True)
    # subtract constant from y_hat_t where non-null
    df_sku_pred_baseline_sale_qty['y_hat_t_sale_qty'] -= hist_baseline_sale_qty_constant


    # margin amt
    df_sku_pred_baseline_margin_amt = pd.DataFrame(np.c_[df_sku_hist_baseline_margin_amt,
                                                         fit_hist_baseline_margin_amt.level,
                                                         fit_hist_baseline_margin_amt.slope,
                                                         fit_hist_baseline_margin_amt.season,
                                                         fit_hist_baseline_margin_amt.fittedvalues],
                                                   columns=['y_t_margin_amt', 'l_t_margin_amt', 'b_t_margin_amt',
                                                            's_t_margin_amt', 'y_hat_t_margin_amt'],
                                                   index=df_sku_hist_baseline_margin_amt.index)

    df_sku_pred_baseline_margin_amt = df_sku_pred_baseline_margin_amt.append(
        fit_hist_baseline_margin_amt.forecast(forward_period).rename('y_hat_t_margin_amt').to_frame(), sort=True)
    # subtract constant from y_hat_t where non-null
    df_sku_pred_baseline_margin_amt['y_hat_t_margin_amt'] -= hist_baseline_margin_amt_constant

#     # Test - visualise the data
#     ax = df_sku_hist_baseline_sale_amt.plot(marker='o', color='black', figsize=(12, 8))
#     df_sku_pred_baseline_sale_amt['y_hat_t_sale_amt'].plot(marker='x', ax=ax, color='blue', legend=True)

#     df_sku_hist_baseline_sale_qty.plot(marker='o', ax=ax, color='green', legend=True)
#     df_sku_pred_baseline_sale_qty['y_hat_t_sale_qty'].plot(marker='x', ax=ax, color='red', legend=True)

#     df_sku_hist_baseline_margin_amt.plot(marker='o', ax=ax, color='yellow', legend=True)
#     df_sku_pred_baseline_margin_amt['y_hat_t_margin_amt'].plot(marker='x', ax=ax, color='cyan', legend=True)
#     plt.show()


    # join all predicted dataframes on index
    sku_pred_baseline_df= pd.merge(df_sku_pred_baseline_sale_amt, df_sku_pred_baseline_sale_qty,
                                   left_index=True, right_index=True)
    sku_pred_baseline_df = pd.merge(sku_pred_baseline_df, df_sku_pred_baseline_margin_amt,
                                    left_index=True, right_index=True)

    logger.debug(f'{sku} - completed prediction of exponential smoothing model...')

    # obtain strength of seasonal and trend components
    # TODO
    
    # finalise baseline and metric table
    sku_pred_baseline_df['date'] = sku_pred_baseline_df.index
    sku_pred_baseline_df['sku_root_id'] = sku

    logger.info(f'{sku} - completed forecast baseline calculation')

    sku_pred_frame.append(sku_pred_baseline_df)
    sku_metric_frame.append(metrics_results_df)
    
if __name__ == "__main__":
    
    
    start_time = time.time()

    logger.info("Loading input tables from Bigquery....")
    
    logger.info("Loading distinct sections table from Bigquery....")
    section_table = load_t0_from_bq(bl_s, project_id)
    
    # Unique sections in category include
    unique_sections = list(section_table["section"].unique())
    logger.info("Unique sections include:")
    for section in unique_sections: logger.info("{a}".format(a=section))
    
    # Loop through sections
    for i_sec in range(0, len(unique_sections)):
        
        section_start_time = time.time()
        section = unique_sections[i_sec]
        
        logger.info("Processing section {a}...".format(a=section))
        
        # Compute the baseline for each section     
        logger.info("Loading summary baseline table from Bigquery....")
        summary_table = load_t1_from_bq(section, project_id)

        logger.info("Computing no. of unique in-scope skus")
        uniq_sku = list(summary_table['sku_root_id'].unique())
        logger.info("No. of in-scope skus: {a}".format(a=len(uniq_sku)))

        # Store the forward looking baseline results and metrics table
        results_df = pd.DataFrame()
        metrics_df = pd.DataFrame()

        # Use the multiproc module to process in parallel
        with Manager() as manager:
            sku_pred_frame = manager.list()  # <-- can be shared between processes.
            sku_metric_frame = manager.list()  # <-- can be shared between processes.
            processes = []

            # Compute the SKU level baseline calculations
            for i in range(0, len(uniq_sku), batchsize):

                # Clear the processes list
                processes[:] = []

                start_time_batch = time.time()
                batch = uniq_sku[i:i+batchsize] # the result might be shorter than batchsize at the end

                for sku in batch:                  
                    p = Process(target=forward_looking_baseline_sku, args=(sku_pred_frame,
                                                                           sku_metric_frame,
                                                                           sku,summary_table, 
                                                                           seasonal_period, 
                                                                           forward_period, 
                                                                           freq, constant))  # Passing the list
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
                output_pred = pd.concat(sku_pred_frame)
                output_metric = pd.concat(sku_metric_frame)
                results_df = pd.concat([results_df, output_pred], ignore_index=True, sort =False)
                results_df.reset_index(drop=True, inplace=True)
                metrics_df = pd.concat([metrics_df, output_metric], ignore_index=True, sort =False)
                metrics_df.reset_index(drop=True, inplace=True)
                sku_pred_frame[:] = [] 
                sku_metric_frame[:] = [] 

                total_time_batch = round((time.time() - start_time_batch), 2)
                logger.debug('Processing with batch size {a} took {b} secs...'.format(a=batchsize, b=total_time_batch))

                logger.info('Results dataframe has {a} rows and {b} cols...'.format(a=results_df.shape[0], b=results_df.shape[1]))


        # Convert all nulls to None
        results_df = results_df.where((pd.notnull(results_df)), None)
        metrics_df = metrics_df.where((pd.notnull(metrics_df)), None)
        
        total_time = round((time.time() - section_start_time) / 60, 1)
        logger.info('Completed baseline processing in {a} mins...'.format(a=total_time))

        # upload the final dataframe onto Bigquery
        logger.info('Uploading forecast baseline table to Bigquery...')
        
        [{'name': 'col1', 'type': 'STRING'},...]
        
        if (i_sec == 0):
            pandas_gbq.to_gbq(results_df, 'baseline_performance.forecast_baseline', project_id=project_id, if_exists=bl_table_config)
            pandas_gbq.to_gbq(metrics_df, 'baseline_performance.forecast_baseline_metrics', project_id=project_id, 
                              if_exists=bl_table_config)
        else:
            pandas_gbq.to_gbq(results_df, 'baseline_performance.forecast_baseline', project_id=project_id, if_exists='append')
            pandas_gbq.to_gbq(metrics_df, 'baseline_performance.forecast_baseline_metrics', project_id=project_id, 
                              if_exists='append')

        logger.info('Completed upload of section forecast baseline to Bigquery...')
    
    
    # Code to append in correct format - to fix issues
    # TODO
#     if (i_sec == 0):
#             pandas_gbq.to_gbq(results_df, 'baseline_performance.forecast_baseline', project_id=project_id, if_exists=bl_table_config, 
#                              table_schema=[{'name': 'b_t_sale_amt', 'type': 'NUMERIC'},
#                                            {'name': 'l_t_sale_amt', 'type': 'NUMERIC'},
#                                            {'name': 's_t_sale_amt', 'type': 'NUMERIC'},
#                                            {'name': 'y_hat_t_sale_amt', 'type': 'NUMERIC'},
#                                            {'name': 'y_t_sale_amt', 'type': 'NUMERIC'},
#                                            {'name': 'b_t_sale_qty', 'type': 'NUMERIC'},
#                                            {'name': 'l_t_sale_qty', 'type': 'NUMERIC'},
#                                            {'name': 's_t_sale_qty', 'type': 'NUMERIC'},
#                                            {'name': 'y_hat_t_sale_qty', 'type': 'NUMERIC'},
#                                            {'name': 'y_t_sale_qty', 'type': 'NUMERIC'},
#                                            {'name': 'b_t_margin_amt', 'type': 'NUMERIC'},
#                                            {'name': 'l_t_margin_amt', 'type': 'NUMERIC'},
#                                            {'name': 's_t_margin_amt', 'type': 'NUMERIC'},
#                                            {'name': 'y_hat_t_margin_amt', 'type': 'NUMERIC'},
#                                            {'name': 'y_t_margin_amt', 'type': 'NUMERIC'},
#                                            {'name': 'date', 'type': 'TIMESTAMP'},
#                                            {'name': 'sku_root_id', 'type': 'STRING'}])
#             pandas_gbq.to_gbq(metrics_df, 'baseline_performance.forecast_baseline_metrics', project_id=project_id, 
#                               if_exists=bl_table_config, table_schema=[{'name': 'alpha', 'type': 'NUMERIC'},
#                                            {'name': 'beta', 'type': 'NUMERIC'},
#                                            {'name': 'phi', 'type': 'NUMERIC'},
#                                            {'name': 'gamma', 'type': 'NUMERIC'},
#                                            {'name': 'l_0', 'type': 'NUMERIC'},
#                                            {'name': 'b_0', 'type': 'NUMERIC'},
#                                            {'name': 'SSE', 'type': 'NUMERIC'},
#                                            {'name': 'MAE', 'type': 'NUMERIC'},
#                                            {'name': 'convergence_flag', 'type': 'INTEGER'},
#                                            {'name': 'metric', 'type': 'STRING'},
#                                            {'name': 'sku_root_id', 'type': 'STRING'}])
#         else:
#             pandas_gbq.to_gbq(results_df, 'baseline_performance.forecast_baseline', project_id=project_id, if_exists='append',
#                              table_schema=[{'name': 'b_t_sale_amt', 'type': 'NUMERIC'},
#                                            {'name': 'l_t_sale_amt', 'type': 'NUMERIC'},
#                                            {'name': 's_t_sale_amt', 'type': 'NUMERIC'},
#                                            {'name': 'y_hat_t_sale_amt', 'type': 'NUMERIC'},
#                                            {'name': 'y_t_sale_amt', 'type': 'NUMERIC'},
#                                            {'name': 'b_t_sale_qty', 'type': 'NUMERIC'},
#                                            {'name': 'l_t_sale_qty', 'type': 'NUMERIC'},
#                                            {'name': 's_t_sale_qty', 'type': 'NUMERIC'},
#                                            {'name': 'y_hat_t_sale_qty', 'type': 'NUMERIC'},
#                                            {'name': 'y_t_sale_qty', 'type': 'NUMERIC'},
#                                            {'name': 'b_t_margin_amt', 'type': 'NUMERIC'},
#                                            {'name': 'l_t_margin_amt', 'type': 'NUMERIC'},
#                                            {'name': 's_t_margin_amt', 'type': 'NUMERIC'},
#                                            {'name': 'y_hat_t_margin_amt', 'type': 'NUMERIC'},
#                                            {'name': 'y_t_margin_amt', 'type': 'NUMERIC'},
#                                            {'name': 'date', 'type': 'TIMESTAMP'},
#                                            {'name': 'sku_root_id', 'type': 'STRING'}])
#             pandas_gbq.to_gbq(metrics_df, 'baseline_performance.forecast_baseline_metrics', project_id=project_id, 
#                               if_exists='append', table_schema=[{'name': 'alpha', 'type': 'NUMERIC'},
#                                            {'name': 'beta', 'type': 'NUMERIC'},
#                                            {'name': 'phi', 'type': 'NUMERIC'},
#                                            {'name': 'gamma', 'type': 'NUMERIC'},
#                                            {'name': 'l_0', 'type': 'NUMERIC'},
#                                            {'name': 'b_0', 'type': 'NUMERIC'},
#                                            {'name': 'SSE', 'type': 'NUMERIC'},
#                                            {'name': 'MAE', 'type': 'NUMERIC'},
#                                            {'name': 'convergence_flag', 'type': 'INTEGER'},
#                                            {'name': 'metric', 'type': 'STRING'},
#                                            {'name': 'sku_root_id', 'type': 'STRING'}])
    
    
    total_time = round((time.time() - start_time) / 60, 1)
    logger.info('Completed forecast baseline processing in {a} mins...'.format(a=total_time))