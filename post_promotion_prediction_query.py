import pandas as pd
import pandas_gbq
import logging
from google.cloud import bigquery

# Set logger properties
logger = logging.getLogger('post_promotion_prediction_model')

def promotion_prediction_(project_id, dataset_id, area):
        
        # Load client
        client = bigquery.Client()

        job_config = bigquery.QueryJobConfig()      
        
        post_promotion_pred_sql = """
        WITH
        temp_aggr_promo AS (
        SELECT
          sku_root_id,
          description,
          area,
          section,
          category,
          subcategory,
          segment,
          brand_name,
          eroskibrand_flag,
          eroskibrand_label,
          wealthy_range_flag,
          flag_healthy,
          innovation_flag,
          tourism_flag,
          local_flag,
          regional_flag,
          no_impacted_stores,
          no_impacted_regions,
          AVG(avg_store_size) AS avg_store_size,
          promo_id,
          promo_year,
          promo_mechanic,
          Promo_mechanic_en,
          name,
          type,
          start_date,
          end_date,
          customer_profile_type,
          marketing_type,
          duration AS duration_days,
          MAX(includes_weekend) AS includes_weekend,
          campaign_start_day,
          campaign_start_month,
          campaign_start_quarter,
          campaign_start_week,
          leaflet_cover,
          leaflet_priv_space,
          in_leaflet_flag,
          in_gondola_flag,
          in_both_leaflet_gondola_flag,
          discount_depth,
          CASE
            WHEN promo_flag_binary =1 THEN 'promotion'
          ELSE
          'post_promotion'
        END
          AS period,
          SUM(p_discount) AS p_discount,
          SUM(p_sale_bl) AS p_sale_bl,
          SUM(p_qty_bl) AS p_qty_bl,
          SUM(p_margin_bl) AS p_margin_bl,
          SUM(p_sale_amt) AS p_sale_amt,
          SUM(p_sale_qty) AS p_sale_qty,
          SUM(p_margin_amt) AS p_margin_amt,
          SUM(p_cal_inc_sale_amt) AS p_cal_inc_sale_amt,
          SUM(p_cal_inc_sale_qty) AS p_cal_inc_sale_qty,
          SUM(p_cal_inc_margin_amt) AS p_cal_inc_margin_amt,
          AVG(pct_inc_sale) AS pct_inc_sale,
          AVG(pct_inc_qty) AS pct_inc_qty,
          AVG(pct_inc_margin) AS pct_inc_margin
        FROM
          `gum-eroski-dev.baseline_performance.baseline_promo`
        WHERE
          promo_mechanic IN ('10',
            '20')
          AND area = "%s"
        GROUP BY
          sku_root_id,
          description,
          area,
          section,
          category,
          subcategory,
          segment,
          brand_name,
          eroskibrand_flag,
          eroskibrand_label,
          wealthy_range_flag,
          flag_healthy,
          innovation_flag,
          tourism_flag,
          local_flag,
          regional_flag,
          no_impacted_stores,
          no_impacted_regions,
          promo_id,
          promo_year,
          promo_mechanic,
          Promo_mechanic_en,
          name,
          type,
          start_date,
          end_date,
          customer_profile_type,
          marketing_type,
          duration,
          campaign_start_day,
          campaign_start_month,
          campaign_start_quarter,
          campaign_start_week,
          leaflet_cover,
          leaflet_priv_space,
          in_leaflet_flag,
          in_gondola_flag,
          in_both_leaflet_gondola_flag,
          discount_depth,
          period ),
        temp_aggr_promo_f AS (
        SELECT
          * EXCEPT (eroskibrand_flag,
            eroskibrand_label,
            wealthy_range_flag),
          CASE
            WHEN eroskibrand_label IS NOT NULL THEN eroskibrand_label
            WHEN wealthy_range_flag = 'N' THEN 'Normal'
            WHEN wealthy_range_flag = 'S' THEN 'Premium'
          ELSE
          NULL
        END
          AS brand_price_label
        FROM
          temp_aggr_promo )
      SELECT
        *
      FROM
        temp_aggr_promo_f
      WHERE
        pct_inc_sale IS NOT NULL
        AND discount_depth IS NOT NULL
        AND promo_mechanic IS NOT NULL
        AND sku_root_id IS NOT NULL
        AND segment IS NOT NULL
        AND period IN ('post_promotion')
      ORDER BY
        sku_root_id,
        promo_id,
        promo_year,
        period
        """ %(area)
         
        # Create a disctionary to loop over all destination tables and scripts
        tables = {'post_prediction_train_input':post_promotion_pred_sql} 
        
        job_config.write_disposition = "WRITE_TRUNCATE"
        for key in tables:
                
                # Set the destination table
                table_ref = client.dataset(dataset_id).table(key)
                job_config.destination = table_ref
     
                # Start the query, passing in the extra configuration.
                query_job = client.query(
                    tables[key],
                    # Location must match that of the dataset(s) referenced in the query
                    # and of the destination table.
                    location='europe-west3',
                    job_config=job_config)  # API request - starts the query

                query_job.result()  # Waits for the query to finish
                logger.info("Completed writing {a} table...".format(a=key))
                
                
def promotion_prediction_res(project_id, dataset_id):
        
        # Load client
        client = bigquery.Client()

        job_config = bigquery.QueryJobConfig()
        
        promo_update = """
        SELECT 
        CAST(p_cal_inc_sale_qty AS NUMERIC) AS p_cal_inc_sale_qty,
        CAST(prediction_interval AS NUMERIC) AS prediction_interval,
        CAST(prediction_error_perc AS NUMERIC) AS prediction_error_perc,
        sku_root_id,description, area, section, category, subcategory, segment,
        brand_name, brand_price_label, flag_healthy, innovation_flag, tourism_flag,
        local_flag, regional_flag, 
        CAST(no_impacted_stores AS INT64) AS no_impacted_stores, 
        CAST(no_impacted_regions AS INT64) AS no_impacted_regions,
        CAST(avg_store_size AS NUMERIC) AS avg_store_size,
        CAST(type AS STRING) AS type,
        customer_profile_type,  marketing_type, 
        CAST(duration_days AS INT64) AS duration_days, 
        includes_weekend, campaign_start_day, 
        campaign_start_month , 
        CAST(campaign_start_quarter AS INT64) AS campaign_start_quarter,
        CAST(campaign_start_week AS INT64) AS campaign_start_week, 
        CAST(leaflet_cover AS INT64) AS leaflet_cover,
        CAST(leaflet_priv_space AS INT64) AS leaflet_priv_space, 
        CAST(in_leaflet_flag AS INT64) AS in_leaflet_flag,
        CAST(in_gondola_flag AS INT64) AS in_gondola_flag,
        CAST(in_both_leaflet_gondola_flag AS INT64) AS in_both_leaflet_gondola_flag,
        CAST(p_qty_bl AS NUMERIC) AS p_qty_bl, 
        promo_mechanic, Promo_mechanic_en , discount_depth, 
        CAST(promoted_in_past AS NUMERIC) as promoted_in_past
        FROM `gum-eroski-dev.prediction_results.prediction_post_promotion_results`
        """

        promotion_pred_sql = """
        WITH res_ini AS (
        SELECT SAFE_DIVIDE(p_cal_inc_sale_qty, p_qty_bl) as perc_uplift_qty, *
        FROM `gum-eroski-dev.prediction_results.prediction_post_promotion_results` 
        WHERE p_qty_bl>50
        ), res_temp AS (
        SELECT *, ROW_NUMBER() 
            over (
                PARTITION BY category 
                order by perc_uplift_qty desc
            ) AS RowNo 
        FROM res_ini 
        )
        SELECT 
        res.RowNo as row_num,
        res.sku_root_id, 
        res.description, 
        res.area, 
        res.section, 
        res.category, 
        res.subcategory, 
        res.segment, 
        res.brand_name, 
        res.brand_price_label,
        res.flag_healthy,
        res.innovation_flag,
        res.tourism_flag,
        res.local_flag,
        res.regional_flag,
        res.promo_mechanic,
        res.Promo_mechanic_en,
        res.discount_depth,
        res.p_qty_bl as baseline_sales_qty,
        res.p_cal_inc_sale_qty as inc_sales_qty,
        res.perc_uplift_qty*100 as perc_uplift_qty,
        res.promoted_in_past
        FROM res_temp res
        WHERE RowNo<=20

        order by category, perc_uplift_qty desc, p_cal_inc_sale_qty desc
        """ 
         
        # Create a disctionary to loop over all destination tables and scripts
        tables = {'prediction_post_promotion_results':promo_update} 
        
        job_config.write_disposition = "WRITE_TRUNCATE"
        for key in tables:
                
                # Set the destination table
                table_ref = client.dataset(dataset_id).table(key)
                job_config.destination = table_ref
     
                # Start the query, passing in the extra configuration.
                query_job = client.query(
                    tables[key],
                    # Location must match that of the dataset(s) referenced in the query
                    # and of the destination table.
                    location='europe-west3',
                    job_config=job_config)  # API request - starts the query

                query_job.result()  # Waits for the query to finish
                logger.info("Completed writing {a} table...".format(a=key))


