import xgboost as xgb
import lightgbm as lgb
#from catboost import CatBoostRegressor, Pool, cv
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import statsmodels.api as sm
import pandas as pd
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import prediction_models

# Set logger properties
logger = logging.getLogger('promotion_prediction_model')
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.FileHandler('promotion_prediction.log')
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

# Scope for the prediction (at an area level)
bl_s = "ALIMENTACION"

# Specify list of input features
# input_features = ['sku_root_id', 'area', 'section', 'category', 'subcategory', 'segment', 'brand_name',
#                   'flag_healthy', 'innovation_flag', 'tourism_flag', 'local_flag', 'regional_flag',
#                   'no_impacted_stores', 'no_impacted_regions', 'avg_store_size', 'Promo_mechanic_en',
#                   'customer_profile_type', 'marketing_type', 'duration_days', 'includes_weekend', 'campaign_start_day',
#                   'campaign_start_month', 'campaign_start_quarter', 'campaign_start_week', 'leaflet_cover',
#                   'leaflet_priv_space', 'in_leaflet_flag', 'in_gondola_flag', 'in_both_leaflet_gondola_flag',
#                   'discount_depth', 'brand_price_label', 'no_hipermercados_stores',	'no_supermercados_stores'
#                   'no_gasolineras_stores',	'no_comercio_electronico_stores',	'no_otros_negocio_stores',
#                   'no_plataformas_stores',	'no_other_stores', 'discount_depth_rank', 'perc_hypermarket']

input_features = ['subcategory', 'segment', 'brand_name',
                  'discount_depth_rank', 'duration_days']

# Specify output features
output_features = ['p_cal_perc_inc_sale_qty']

# Specify exclusion months
test_months_exclusion = ['Jan', 'Aug', 'Nov', 'Dec']

# Specify promo mechanics
mechanic = [10, 20]

# Specify the min no. of impacted stores
impact_stores_outlier = 1

# Specify the max promo duration
promo_duration_outlier = 40

discount_depths_outlier = ['2.5% off','5% off','10% off','15% off','20% off','25% off','30% off','35% off','40% off','45% off',
                           '50% off','55% off','60% off','buy 2 pay 1','buy 2 pay 1.2',
                           'buy 2 pay 1.3','buy 2 pay 1.4','buy 2 pay 1.5','buy 2 pay 1.6','buy 2 pay 1.7','buy 2 pay 1.8','buy 3 pay 2',
                           'buy 4 pay 3']

# Specify train, test, forecast
run_config = 'train' # values include 'train', 'train-predict', 'forecast'

# Specify categorical cols
cat_columns = ['sku_root_id', 'description', 'segment', 'subcategory', 'category', 'section', 'area', 'brand_name', 'flag_healthy',
               'innovation_flag', 'tourism_flag', 'local_flag', 'regional_flag', 'Promo_mechanic_en','promo_mechanic','name',
               'start_date', 'end_date', 'customer_profile_type',
               'marketing_type', 'includes_weekend', 'campaign_start_day', 'campaign_start_month',
               'campaign_start_quarter',
               'campaign_start_week', 'leaflet_cover', 'leaflet_priv_space', 'in_leaflet_flag', 'in_gondola_flag',
               'in_both_leaflet_gondola_flag', 'discount_depth', 'period', 'brand_price_label', 'type', 'promo_id', 'promo_year']


def run_regression_model_single(input_data_sku_list, discount_depth_list, train_model, join_fields, mapping_dict):

    logger.info("Running regression prediction...")

    # Take the cartesian product of the input data and discount depth list
    input_data_sku_list['m-key'] = 1
    discount_depth_list['m-key'] = 1
    input_data_sku_list = input_data_sku_list.merge(discount_depth_list, on='m-key', how='inner')
    input_data_sku_list.drop(['m-key'], axis=1, inplace=True)

    # Compute additional interaction term fields - to do

    # Takes the input data sku list (for in-scope categories) and does an inner join with the train model
    input_data_sku_list = input_data_sku_list.merge(train_model, on=join_fields, how='inner')

    logger.info("Sample data includes {b} samples to predict...".format(b=input_data_sku_list.shape[0]))


    # Select only the relevant fields from  input_data_sku_list
    features = input_features + output_features
    features = list(set(features) - set(join_fields))
    pred_features = list(set(features) - set(output_features))
    input_data_sku_list_apply = input_data_sku_list.copy()

    # Map fields using the mapping dict
    if len(mapping_dict) != 0:
        logger.info("Applying mapping to sample data...")
        # Apply mapping on the items in X_apply
        for col in mapping_dict:
            if col in list(input_data_sku_list.columns):
                # apply mapping - any new values not in mapping will be set to NaN
                unique_vals_dict = mapping_dict[col]  #
                input_data_sku_list_apply[col] = input_data_sku_list_apply[col].map(unique_vals_dict)

    # Filter on relevant features
    # input_data_sku_list = input_data_sku_list[features]
    return input_data_sku_list_apply
    # Loop through rows of input data and apply the model
    # results_df = pd.DataFrame()
    # logger.info("Applying model to predict values for each sku and promotional mechanic permutation...")
    #
    # for index, row in input_data_sku_list_apply.iterrows():
    #
    #     logger.info("Predicting results for permutation {a} of {b}".format(a=index, b=input_data_sku_list_apply.shape[0]))
    #     # model
    #     model = row['model']
    #
    #     X_pred = row[pred_features]
    #     X_pred_df = X_pred.to_frame().T
    #     X_pred_df.reset_index(drop=True, inplace=True)
    #     X_pred_df = sm.add_constant(X_pred_df, has_constant='add')
    #
    #     y_pred = model.predict(X_pred_df)  # predict out of sample
    #
    #     # save the results
    #     row_df = row.to_frame().T
    #     row_df.reset_index(drop=True, inplace=True)
    #     row_df['y_pred'] = y_pred[0]
    #     results_df = pd.concat([results_df, row_df], axis=1)
    #
    # # save the results
    # logger.info("Completed prediction of target variable...")
    # return results_df


if __name__ == "__main__":

    start_time = time.time()

    # load csv input file
    logger.info("Loading historical promotion performance input data...")
    input_data = pd.read_csv("C:\\Users\\hamzajuzer\\PycharmProjects\\prediction\\promotion_input_data.csv", sep=",",
                             encoding='latin1')

    start_time = time.time()

    # Train the model
    if run_config == 'train' or run_config == 'train-predict':
        logger.info("Training the prediction model for the promotion period...")

        # train ML model
        train_model, map_dict, mae, mape, join_fields, filtered_model = train_promotion_prediction_model(input_data,
                                                                                                         input_features,
                                                                                                         output_features,
                                                                                                         cat_columns,
                                                                                                         model='regression',
                                                                                                         # either lightgbm, xgboost, catboost or regression
                                                                                                         learning_rate=0.03,  # set between 0.01-0.05
                                                                                                         max_depth=200,
                                                                                                         # 100 for lightgbm, 50 for xgboost
                                                                                                         num_leaves=250,  # for lightgbm
                                                                                                         n_iter=10000,
                                                                                                         # for lightgbm, no. of iterations, 20000
                                                                                                         n_estimators=150,
                                                                                                         # for xgboost, no of estimators
                                                                                                         train_size=0.8,  # test train split
                                                                                                         test_months_exclusion=None,
                                                                                                         # exclude certain months
                                                                                                         cat_var_exclusion=False,
                                                                                                         # exclude specification of categorical variables (lightgbm)
                                                                                                         remove_outliers=True,
                                                                                                         impact_stores_outlier=impact_stores_outlier,
                                                                                                         promo_duration_outlier=promo_duration_outlier,
                                                                                                         discount_depths_outlier=discount_depths_outlier)  # remove outliers


        # save the train model as csv
        train_model.to_csv("train_model.csv", encoding='utf-8', index=False)
        filtered_model.to_csv("train_input.csv", encoding='utf-8', index=False)


        # predict
        # load csv input file
        logger.info("Loading prediction input data...")
        input_data_sku_list = pd.read_csv("C:\\Users\\hamzajuzer\\PycharmProjects\\prediction\\in_scope_skus_full.csv",
                                 sep=",",
                                 encoding='latin1')

        discount_depth_list = pd.read_csv("C:\\Users\\hamzajuzer\\PycharmProjects\\prediction\\discount_depth.csv",
                                 sep=",",
                                 encoding='latin1')

        results_df = run_regression_model_single(input_data_sku_list, discount_depth_list, train_model, join_fields,
                                                 map_dict)

        # save the train model as csv
        results_df.to_csv("results.csv", encoding='utf-8', index=False)

    total_time = round((time.time() - start_time) / 60, 1)
    logger.info('Completed ML processing in {a} mins...'.format(a=total_time))

