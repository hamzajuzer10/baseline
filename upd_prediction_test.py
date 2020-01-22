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

# Set logger properties
logger = logging.getLogger('promotion_prediction_model')
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.FileHandler('promotion_prediction.log')
fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

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

input_features = ['subcategory', 'segment', 'brand_name' ,'category',
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

def plotScatter(df, xvar, yvar, marker):

    # Use the 'hue' argument to provide a factor variable
    sns.lmplot(x=xvar, y=yvar, data=df, fit_reg=False, hue=marker, legend=False)

    # Move the legend to an empty part of the plot
    plt.legend(loc='lower right')

    plt.title('Promotional discount relationship')
    plt.tight_layout()
    plt.show()

def plotImp(model, train_model, X , num = 20):

    if model == 'lightgbm':
        feature_imp = pd.DataFrame({'Value':train_model.feature_importance(),'Feature': X.columns})
    elif model == 'xgboost':
        feature_imp = pd.DataFrame({'Value':train_model.feature_importances_,'Feature': X.columns})

    plt.figure(figsize=(10, 10))
    sns.set(font_scale = 1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                        ascending=False)[0:num])
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()


def plothist(y_validation, pred):
    plt.subplot(1, 2, 1)
    n, bins, patches = plt.hist(y_validation[output_features[0]].values - pred, 10000, facecolor='red', alpha=0.75)
    plt.xlabel('error (uplift units) xgboost')
    plt.xlim(-10000, 10000)
    plt.title('Histogram of error')

    plt.subplot(1, 2, 2)
    n, bins, patches = plt.hist(y_validation[output_features[0]].values, 10000, facecolor='blue', alpha=0.75)
    plt.xlabel('Error (naive - 0 units uplift)')
    plt.xlim(-10000, 10000)
    plt.title('Histogram of error')
    plt.tight_layout()
    plt.show()


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

def train_promotion_prediction_model(input_data, input_features, cat_columns, model, learning_rate, max_depth,
                                     num_leaves, n_iter, n_estimators,
                                     train_size, test_months_exclusion, cat_var_exclusion,
                                     remove_outliers, impact_stores_outlier=None, promo_duration_outlier=None,
                                     discount_depths_outlier=None):
    """train the promotion model during the promotion weeks
    :Args
        X(dataframe): dataframe containing the input data to train the model
        y(dataframe): dataframe containing the output values to train the model on
        input_features(list): list of cols that we will be using
        cat_columns(list): list of cols that are categorical
        learning_rate(float): step size shrinkage used to prevent overfitting. Range is [0,1]
        max_depth(int): determines how deeply each tree is allowed to grow during any boosting round
        subsample(float): percentage of samples used per tree. Low value can lead to underfitting
        colsample_bytree(float): percentage of features used per tree. High value can lead to overfitting
        n_iter(int): number of iterations you want to run
        train_size_perc(float): train test splits
        test_months_exclusion: identifies the months to be excluded from the training data set
    :return:
        xgboost model(model): xgboost ML model
    """

    # convert input data format
    # for cols in input data that is not in the cat cols list, convert to numeric
    for col in list(input_data.columns):
        if col not in list(cat_columns):
            input_data[col] = pd.to_numeric(input_data[col])

    # Check no. of input sample data rows
    logger.info("Input sample data includes {b} samples...".format(b=input_data.shape[0]))

    # Lets remove data within the test exclusion months list
    if 'campaign_start_month' in list(input_data.columns) and test_months_exclusion is not None:
        outliers = input_data[input_data.campaign_start_month.isin(test_months_exclusion)]
        logger.info(
            "Removing sample data where campaign start months is in {a}, {b} sample data points removed...".format(
                a=test_months_exclusion,
                b=outliers.shape[0]))
        input_data = input_data[~input_data['campaign_start_month'].isin(test_months_exclusion)]


    # Lets remove data where store count is below a certain value
    if 'no_impacted_stores' in list(input_data.columns) and impact_stores_outlier is not None:
        outliers = input_data[input_data['no_impacted_stores'] < impact_stores_outlier]
        logger.info("Removing sample data where impacted stores < {a}, {b} sample data points removed...".format(
            a=impact_stores_outlier,
            b=outliers.shape[0]))
        input_data = input_data[input_data['no_impacted_stores'] >= impact_stores_outlier]

    # Lets remove data where duration is above a certain value
    if 'duration_days' in list(input_data.columns) and promo_duration_outlier is not None:
        outliers = input_data[input_data['duration_days'] > promo_duration_outlier]
        logger.info("Removing sample data where promotion duration > {a}, {b} sample data points removed...".format(
            a=promo_duration_outlier,
            b=outliers.shape[0]))
        input_data = input_data[input_data['duration_days'] <= promo_duration_outlier]

    # Lets remove data where discount depth is not in specified list
    if 'discount_depth' in list(input_data.columns) and discount_depths_outlier is not None:
        outliers = input_data[~input_data.discount_depth.isin(discount_depths_outlier)]
        logger.info("Removing sample data where discount depth is not in {a}, {b} sample data points removed...".format(
            a=discount_depths_outlier,
            b=outliers.shape[0]))
        input_data = input_data[input_data.discount_depth.isin(discount_depths_outlier)]

    if remove_outliers:
        logger.info("Removing outliers from sample data...")

        # outlier removal based on negative values
        outliers = input_data[input_data[output_features[0]] <= 0]
        logger.info(
            "Removing all negative values from {a}, {b} sample data points removed...".format(a=output_features[0],
                                                                                              b=outliers.shape[0]))

        input_data = input_data[input_data[output_features[0]] > 0]

        # outlier removal based on too high % uplift - set value to 1000%
        if 'p_cal_perc_inc_sale_qty' in list(input_data.columns):
            outliers = input_data[input_data['p_cal_perc_inc_sale_qty'] >= 10]
            logger.info(
                "Removing sample data where % qty uplift is greater than 1000%, {b} sample data points removed...".format(
                    b=outliers.shape[0]))
            input_data = input_data[input_data['p_cal_perc_inc_sale_qty'] < 10]

        # outlier removal based on quantile in target variable
        q = input_data[output_features[0]].quantile(0.95)

        outliers = input_data[input_data[output_features[0]] >= q]
        logger.info("Based on 95% quantiles, {} sample data points removed...".format(outliers.shape[0]))

        input_data = input_data[input_data[output_features[0]] < q]

    # Filter on only the input features
    total_features = input_features + output_features
    input_data = input_data[total_features]

    # Check absent values
    null_value_stats_x = input_data[input_features].isnull().sum(axis=0)
    logger.info("Null values for input features include:\n{}".format(null_value_stats_x[null_value_stats_x != 0]))

    null_value_stats_y = input_data[output_features].isnull().sum(axis=0)
    logger.info("Null values for target variable include:\n{}".format(null_value_stats_y[null_value_stats_y != 0]))

    # Throw error if any values are null in y
    if input_data[output_features].isnull().values.any():
        logger.error("Null values found in target data...")
        raise ValueError('Null values found in target data!')

    # Fill remaining absent values in X with -999
    input_data.fillna(-999, inplace=True)

    # Describe the dataset
    logger.info("Summary statistics for numeric features in input data are...")
    logger.info("{}".format(input_data.describe()))

    #print(input_data.describe())

    # Check data types
    X = input_data[input_features]
    y = input_data[output_features]*100
    logger.info("Input dataset data types include:\n{}".format(X.dtypes))
    logger.info("Target variable data types include:\n{}".format(y.dtypes))

    # Lets split the data into training and validation sets
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=train_size, random_state=42)
    logger.info("Training dataset includes {} samples...".format(X_train.shape[0]))
    logger.info("Test dataset includes {} samples...".format(X_validation.shape[0]))

    # create a mapping dictionary (to be used for models which require int categorical cols)
    map_dict = {}

    if model == 'CatBoost':

        # using the catboost model
        params = {'depth': [4, 7, 10],
                  'learning_rate': [0.03, 0.1, 0.15],
                  'l2_leaf_reg': [1, 4, 9],
                  'iterations': [300]}

        # Obtain categorical feature index
        cat_features_index = [X.columns.get_loc(c) for c in cat_columns if c in X]

        # initialise CatBoost regressor
        train_model = CatBoostRegressor(iterations=700,
                                        learning_rate=learning_rate,
                                        depth=max_depth,
                                        eval_metric='RMSE',
                                        random_seed=42,
                                        bagging_temperature=0.2,
                                        od_type='Iter',
                                        metric_period=75,
                                        od_wait=100)

        # Fit the model - catboost does not require us to specify integers for cat features
        train_model.fit(X_train, y_train,
                        eval_set=(X_validation, y_validation),
                        cat_features=cat_features_index,
                        use_best_model=True)

        # Calculate feature importance
        fea_imp = pd.DataFrame({'imp': train_model.feature_importances_, 'col': X.columns})
        fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False])
        #         fea_imp.plot(kind='barh', x='col', y='imp', figsize=(10, 7), legend=None)
        #         plt.title('CatBoost - Feature Importance')
        #         plt.ylabel('Features')
        #         plt.xlabel('Importance');

        # Save model
        train_model.save_model('catboost_model.dump')

    elif model == 'lightgbm':

        # For lightgbm, we need to convert our categorical features to int
        # Loop through categorical cols
        for col in cat_columns:
            if col in list(X.columns):
                # get unique values
                unique_vals = X[col].unique()
                unique_vals_dict = dict([(val, num) for num, val in enumerate(unique_vals)])

                # map them for the train and test data sets
                X_train = X_train.copy()
                X_train[col] = X_train[col].map(unique_vals_dict)
                X_validation = X_validation.copy()
                X_validation[col] = X_validation[col].map(unique_vals_dict)

                # store the mapping for later use
                map_dict[col] = unique_vals_dict

        # LightGBM dataset formatting (with categorical variables)
        if cat_var_exclusion:
            lgtrain = lgb.Dataset(X_train, y_train,
                                  feature_name=input_features)
            lgvalid = lgb.Dataset(X_validation, y_validation,
                                  feature_name=input_features)
        else:
            cat_col = [col for col in cat_columns if col in list(X.columns)]

            lgtrain = lgb.Dataset(X_train, y_train,
                                  feature_name=input_features,
                                  categorical_feature=cat_col)
            lgvalid = lgb.Dataset(X_validation, y_validation,
                                  feature_name=input_features,
                                  categorical_feature=cat_col)

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'boosting_type': 'gbdt',
            'verbosity': -1
        }

        train_model = lgb.train(
            params,
            lgtrain,
            num_boost_round=n_iter,
            valid_sets=[lgtrain, lgvalid],
            valid_names=["train", "valid"],
            early_stopping_rounds=1000,
            verbose_eval=500
        )

        pred = train_model.predict(X_validation)

    elif model == 'xgboost':

        # For xgboost, we need to convert our categorical features to int
        # There are 3 approaches - one-hot encode, label encode and binary encode

        # Here, for simplicity, we are using label encoders
        # Loop through categorical cols
        for col in cat_columns:
            if col in list(X.columns):
                # get unique values
                unique_vals = X[col].unique()
                unique_vals_dict = dict([(val, num) for num, val in enumerate(unique_vals)])

                # map them for the train and test data sets
                X_train = X_train.copy()
                X_train[col] = X_train[col].map(unique_vals_dict)
                X_validation = X_validation.copy()
                X_validation[col] = X_validation[col].map(unique_vals_dict)

                # store the mapping for later use
                map_dict[col] = unique_vals_dict

        train_model = xgb.XGBRegressor(objective='reg:linear',
                                       colsample_bytree=0.3,
                                       learning_rate=learning_rate,
                                       max_depth=max_depth,
                                       alpha=10,
                                       n_estimators=n_estimators,
                                       verbosity=2)

        train_model.fit(X_train, y_train)

        pred = train_model.predict(X_validation)

    elif model == 'regression':

        # Use linear regression only if subcategory and brand name and segment is included in the list
        # Compute linear regression coefficients for the following combinations
        # 1) Subcategory and brand
        # 2) Segment
        # 3) Subcategory

        # Primary regressor variable includes discount_depth_rank

        # Compute coefficients for the remaining fields in the input data set
        # Output the R^2, p_value, and stdev for each combination
        # Follow a hierarchy when applying the model to each sku
        # If the subcategory and brand the sku sits in has an R^2, p_value and stdev smaller/ larger than a given threshold,
        # use, the segment model and likewise, when the segment model has an R^2, p_value and stdev smaller/ larger than a given
        # threshold, use the subcategory model

        # We will only thus be able to predict values for segments/ categories and subcategories where there has been a
        # promotion in the past
        if ('category' not in list(input_data[input_features].columns)) or \
                ('brand_name' not in list(input_data[input_features].columns)) or \
                ('segment' not in list(input_data[input_features].columns)) or \
                ('discount_depth_rank' not in list(input_data[input_features].columns)):
            logger.error(
                "Currently performing a linear regression per subcategory and/ or brand and/ or segment with discount depth rank "
                "as the primary regressor. However subcategory and brand name and segment and discount depth rank is not defined as "
                "an input variable!")
            raise ValueError('Subcategory and brand name and segment and discount depth rank is not defined as an input variable')

        # For simplicity, use all data to train the model and compute the R2, stdev, intercept and coefficient
        logger.info("For regression, both train and test datasets will be used to train the model...")
        logger.info("Combined sample dataset includes {} samples...".format(input_data.shape[0]))

        # Loop through each combination and compute the regression
        combination = {('category', 'brand_name'): 1, ('segment',): 2, ('subcategory',): 3}
        all_combination = ('category', 'brand_name', 'segment' ,'subcategory')

        # Convert all categorical variables into numeric for regression
        input_data_train = input_data.copy()
        for col in cat_columns:
            if col in list(input_data.columns) and col not in all_combination:
                # get unique values
                unique_vals = input_data[col].unique()
                unique_vals_dict = dict([(val, num) for num, val in enumerate(unique_vals)])

                # map the input dataset
                input_data_train[col] = input_data_train[col].map(unique_vals_dict)

                # store the mapping for later use
                map_dict[col] = unique_vals_dict

        # Create a dataframe to store the results
        train_model_all = pd.DataFrame(columns=['rank', 'agg_fields', 'key', 'model', 'no_data_points', 'outlier_data_points',
                                            'r2', 'rmse', 'mean_error', 'mae', 'mape'])

        # Create a dataframe to store the validation set to compute overall metrics
        valid_model = pd.DataFrame()
        filtered_model = pd.DataFrame()


        for agg_list in combination.keys():

            # Training model for combination
            logger.info("Training linear regression model for {a}...".format(a=agg_list))

            # get unique values of both subcat and brand
            unique_df = input_data_train.drop_duplicates(list(agg_list))[list(agg_list)].reset_index(drop=True)
            logger.info("There are {a} unique {b} in the data...".format(a=unique_df.shape[0], b=agg_list))

            # group by agg_list
            input_data_model = input_data_train.groupby(list(agg_list))

            # Select the list of input attributes not in agg_list
            training_list = list(set(list(input_data_train[input_features].columns)) - set(all_combination))
            logger.debug("Training features include {}".format(training_list))

            for key, group in input_data_model:

                # Convert key to tuple if not
                if not isinstance(key, tuple):
                    key_t = (key,)
                else:
                    key_t = key

                # Train the model for each group
                logger.info("Training linear regression model for {a}...".format(a=key_t))
                logger.info("There are {a} data samples...".format(a=group.shape[0]))

                n_data_points = group.shape[0]

                # Lets remove all outlier data points with high z scores
                q_group = group[output_features[0]].quantile(0.95)

                outliers = group[group[output_features[0]] >= q_group]
                logger.info("Removing outlier data points...")
                logger.info("Based on 95% quantiles, {} sample data points removed...".format(outliers.shape[0]))

                outlier_data_points = outliers.shape[0]

                group = group[group[output_features[0]] < q_group]

                # If there is less than 3 sample data points, then skip
                if group.shape[0] < 3:
                    logger.info("Too few data sample needed for training...")
                    logger.info("Skipping to next group....")

                    train_model_dict = {'rank': combination[agg_list],
                                        'agg_fields': agg_list,
                                        'key': key_t,
                                        'model': None,
                                        'no_data_points': 0,
                                        'outlier_data_points': outlier_data_points,
                                        'r2': None,
                                        'rmse': None,
                                        'mean_error': None,
                                        'mae': None,
                                        'mape': None}

                    # add train model to dataframe
                    train_model_all = train_model_all.append(train_model_dict, ignore_index=True)
                    continue

                # Append group to validation dataset
                v_group = group[list(all_combination)].copy()
                valid_model = valid_model.append(v_group.drop_duplicates())
                filtered_model = filtered_model.append(group)
                n_data_points_used = group.shape[0]


                t_list = training_list+output_features
                group = group[t_list]

                # plot the relationship
                # plotScatter(group, "discount_depth_rank", output_features[0], "Promo_mechanic_en")

                X_reg = group[training_list]

                # force to add constant if a constant values is already supplied
                # this will ensure consistency in output format
                X_reg = sm.add_constant(X_reg, has_constant='add')
                y_reg = group[output_features]

                # Train robust linear regression model
                reg_model = sm.OLS(y_reg, X_reg).fit()
                logger.info("Completed model training...")


                # Log the model results
                logger.debug("Model results...")
                logger.debug("\n{}".format(reg_model.summary()))

                mape = np.divide(reg_model.resid.values,
                                 y_reg[output_features[0]].values)
                mape[mape == np.inf] = 0
                mape[mape == -np.inf] = 0
                mape = np.median(np.abs(mape))

                train_model_dict = {'rank': combination[agg_list],
                                'agg_fields': agg_list,
                                'key': key_t,
                                'model': reg_model,
                                'no_data_points': n_data_points_used,
                                'outlier_data_points': outlier_data_points,
                                'r2': reg_model.rsquared,
                                'rmse': np.sqrt(np.mean(np.square(reg_model.resid.values))),
                                'mean_error': np.mean(reg_model.resid.values),
                                'mae': np.median(np.abs(reg_model.resid.values)),
                                'mape': mape}

                # add train model to dataframe
                train_model_all = train_model_all.append(train_model_dict, ignore_index=True)


    if model in ('catboost', 'lightgbm', 'xgboost'):

        # Evaluate the model
        rmse = np.sqrt(mean_squared_error(y_validation, pred))
        logger.info("RMSE: {}".format(rmse))

        mean_error = np.mean(y_validation[output_features[0]].values - pred)
        logger.info("Mean Error: {}".format(mean_error))

        mae = np.median(np.absolute(y_validation[output_features[0]].values - pred))
        logger.info("MAE: {}".format(mae))

        mape = np.divide(y_validation[output_features[0]].values - pred, y_validation[output_features[0]].values)
        mape[mape == np.inf] = 0
        mape[mape == -np.inf] = 0
        mape = np.median(np.abs(mape))
        logger.info("MAPE: {}%".format(mape * 100))

        val_std = np.std(y_validation[output_features[0]].values)
        logger.info("Benchmark STD: {}".format(val_std))

        val_mean = np.mean(y_validation[output_features[0]].values)
        logger.info("Benchmark Mean Error: {}".format(val_mean))

        val_mae = np.mean(np.absolute(y_validation[output_features[0]].values))
        logger.info("Benchmark MAE: {}".format(val_mae))

        logger.info("Benchmark MAPE: -100%")

        # plot the results
        plothist(y_validation, pred)

        # plot the feature importance
        plotImp(model, train_model, X, num=20)

        join_fields = None
        filtered_model = None


    elif model in ('regression'):

        # get the unique values in the validation model
        valid_model = valid_model.drop_duplicates(list(all_combination))[list(all_combination)].reset_index(drop=True)
        valid_model['p_id'] = valid_model[list(all_combination)].apply(tuple, axis=1)

        # For each line in valid_model, find the corresponding rows in train model where key is in p_id
        train_model = pd.DataFrame()
        logger.info("Computing best model for each {a}".format(a=all_combination))
        for index, row in valid_model.iterrows():

            # logging progress
            logger.info("{a} out of {b} permutations complete...".format(a=index, b=valid_model.shape[0]))

            # find all the rows in train model that satisfy the criterion
            train_model_all['valid'] = train_model_all.apply(lambda x: set(x.key).issubset(row.p_id), axis=1)

            # filter on the rows that are true
            valid_rows = train_model_all[train_model_all['valid'] == True]

            # Using the rank condition, lets filter on only the model which has the most favorable property
            # R2 threshold > 0.2 and
            if len(valid_rows[(valid_rows['r2'] >= 0.3)]) >= 1:
                valid_rows = valid_rows[valid_rows['r2'] >= 0.2]

            valid_rows = valid_rows.sort_values(by='mape', ascending=True).head(1)

            # Include the valid model all combinations cols
            valid_rows.reset_index(drop=True, inplace=True)
            row_df = row.to_frame().T
            row_df.reset_index(drop=True, inplace=True)

            # Include the model coefficient values
            coeff_df = valid_rows['model'][0].params.to_frame().T
            coeff_df.reset_index(drop=True, inplace=True)
            coeff_df = coeff_df.add_prefix('coeff_')

            valid_rows = pd.concat([valid_rows, row_df], axis=1)
            valid_rows = pd.concat([valid_rows, coeff_df], axis=1)


            train_model = train_model.append(valid_rows)

        # Compute aggregate statistics
        rmse = train_model['rmse'].mean()
        logger.info("RMSE: {}".format(rmse))

        mean_error = train_model['mean_error'].mean()
        logger.info("Mean Error: {}".format(mean_error))

        mae = train_model['mae'].median()
        logger.info("MAE: {}".format(mae))

        mape = train_model['mape'].median()
        logger.info("MAPE: {}%".format(mape * 100))

        mae_std = train_model['mae'].std()
        logger.info("MAE_std: {}".format(mae_std))

        mape_std = train_model['mape'].std()
        logger.info("MAPE_std: {}%".format(mape_std * 100))

        R2_avg = train_model['r2'].mean()
        logger.info("R^2_avg: {}".format(R2_avg))

        join_fields = list(all_combination)

    return train_model, map_dict, mae, mape, join_fields, filtered_model


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
        train_model, map_dict, mae, mape, join_fields, filtered_model = train_promotion_prediction_model(input_data, input_features, cat_columns,
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


        # # predict
        # # load csv input file
        # logger.info("Loading prediction input data...")
        # input_data_sku_list = pd.read_csv("C:\\Users\\hamzajuzer\\PycharmProjects\\prediction\\in_scope_skus_full.csv",
        #                          sep=",",
        #                          encoding='latin1')
        #
        # discount_depth_list = pd.read_csv("C:\\Users\\hamzajuzer\\PycharmProjects\\prediction\\discount_depth.csv",
        #                          sep=",",
        #                          encoding='latin1')
        #
        # results_df = run_regression_model_single(input_data_sku_list, discount_depth_list, train_model, join_fields,
        #                                          map_dict)
        #
        # # save the train model as csv
        # results_df.to_csv("results.csv", encoding='utf-8', index=False)

    total_time = round((time.time() - start_time) / 60, 1)
    logger.info('Completed ML processing in {a} mins...'.format(a=total_time))

