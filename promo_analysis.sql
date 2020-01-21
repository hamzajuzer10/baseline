--Create div period table
CREATE OR REPLACE TABLE
`gum-eroski-dev.promo_analysis.sku_store_div_period` AS
WITH rank_start AS 
(SELECT sku_root_id, store_id, date, 
RANK() OVER (
    PARTITION BY sku_root_id, store_id
    ORDER BY date asc) as start_date_trans_rank
FROM `gum-eroski-dev.promo_analysis.aggregate_daily_transaction_to_sku_bline` 
where total_sale_qty <> 0 
), top_start AS (
SELECT * FROM rank_start
WHERE start_date_trans_rank=1
), rank_end AS (
SELECT sku_root_id, store_id, date, 
RANK() OVER (
    PARTITION BY sku_root_id, store_id
    ORDER BY date desc) as end_date_trans_rank
FROM `gum-eroski-dev.promo_analysis.aggregate_daily_transaction_to_sku_bline` 
where total_sale_qty <> 0 
), top_end AS (
SELECT * FROM rank_end
WHERE end_date_trans_rank=1
)
SELECT 
top_start.sku_root_id,
top_start.store_id,
top_start.date as start_date,
top_end.date as end_date,
DATE_DIFF(top_end.date, top_start.date, DAY)+1 AS period
from top_start
LEFT JOIN top_end
on top_end.sku_root_id = top_start.sku_root_id
and top_end.store_id = top_start.store_id
order by sku_root_id, store_id;

---Create a daily baseline average
CREATE OR REPLACE TABLE
`gum-eroski-dev.promo_analysis.daily_sku_store_sale_average` AS
WITH sum_sku_store AS (
SELECT sku_root_id, store_id, sum(total_sale_qty) as total_sale_qty
FROM `gum-eroski-dev.promo_analysis.aggregate_daily_transaction_to_sku_bline` 
GROUP BY sku_root_id, store_id
)
SELECT sum_sku_store.sku_root_id, 
sum_sku_store.store_id, 
SAFE_DIVIDE(sum_sku_store.total_sale_qty, trans_period.period) AS avg_sale_qty,
trans_period.period
FROM sum_sku_store
INNER JOIN `gum-eroski-dev.promo_analysis.sku_store_div_period` trans_period 
on trans_period.sku_root_id = sum_sku_store.sku_root_id
and trans_period.store_id = sum_sku_store.store_id;

--Create a day of week daily baseline average
CREATE OR REPLACE TABLE
  `promo_analysis.daily_sku_store_sale_average_dayweek`
AS   
with table_days as (
select sku_root_id,store_id,date,FORMAT_DATE('%A',DATE) as dayweek, total_sale_qty from `promo_analysis.aggregate_daily_transaction_to_sku_bline` 
),
jointable as(
select a.*, b.start_date, b.end_date, b.period from table_days a
inner join `promo_analysis.sku_store_div_period`  b
on a.sku_root_id=b.sku_root_id and 
a.store_id=b.store_id and 
a.date>=b.start_date and 
a.date<=b.end_date)
select sku_root_id,store_id,dayweek, avg(total_sale_qty) as total_sale_qty, max(period) as period 
from jointable
group by sku_root_id,store_id,dayweek;

--Update daily aggregate promo to sku bline table
CREATE OR REPLACE TABLE
`gum-eroski-dev.promo_analysis.daily_total_aggregate_promo_to_sku_bline` 
PARTITION BY 
date
CLUSTER BY
promo_id,store_id,sku_root_id,promo_mechanic
AS
SELECT promo.* EXCEPT(bline_qty_all, bline_qty_all_day),
IFNULL(bline_avg.avg_sale_qty,0) as bline_qty_all,
IFNULL(bline_avg_day.total_sale_qty,0) as bline_qty_all_day
FROM `gum-eroski-dev.promo_analysis.daily_total_aggregate_promo_to_sku_bline` promo
left join `gum-eroski-dev.promo_analysis.daily_sku_store_sale_average` bline_avg
on bline_avg.sku_root_id = promo.sku_root_id
and bline_avg.store_id = promo.store_id
left join `gum-eroski-dev.promo_analysis.daily_sku_store_sale_average_dayweek` bline_avg_day
on bline_avg_day.sku_root_id = promo.sku_root_id
and bline_avg_day.store_id = promo.store_id
and bline_avg_day.dayweek = FORMAT_DATE('%A',promo.date) ;


--Update the daily aggregate promo to sku pull forward table
CREATE OR REPLACE TABLE
`gum-eroski-dev.promo_analysis.daily_total_aggregate_promo_to_sku_pf` 
PARTITION BY 
date
CLUSTER BY
promo_id,store_id,sku_root_id,promo_mechanic
AS
SELECT promo.*,
IFNULL(bline_avg.avg_sale_qty,0) as bline_qty_all,
IFNULL(bline_avg_day.total_sale_qty,0) as bline_qty_all_day
FROM `gum-eroski-dev.promo_analysis.daily_total_aggregate_promo_to_sku_pf` promo
left join `gum-eroski-dev.promo_analysis.daily_sku_store_sale_average` bline_avg
on bline_avg.sku_root_id = promo.sku_root_id
and bline_avg.store_id = promo.store_id
left join `gum-eroski-dev.promo_analysis.daily_sku_store_sale_average_dayweek` bline_avg_day
on bline_avg_day.sku_root_id = promo.sku_root_id
and bline_avg_day.store_id = promo.store_id
and bline_avg_day.dayweek = FORMAT_DATE('%A',promo.date) ;

--Update the daily transaction to sku bline table
CREATE OR REPLACE TABLE
`gum-eroski-dev.promo_analysis.aggregate_daily_transaction_to_sku_bline`  
PARTITION BY 
date
CLUSTER BY
store_id,sku_root_id,area,section
AS
SELECT trans.* EXCEPT(bline_qty_all, bline_qty_all_day),
IFNULL(bline_avg.avg_sale_qty,0) as bline_qty_all,
IFNULL(bline_avg_day.total_sale_qty,0) as bline_qty_all_day
FROM `gum-eroski-dev.promo_analysis.aggregate_daily_transaction_to_sku_bline` trans
left join `gum-eroski-dev.promo_analysis.daily_sku_store_sale_average` bline_avg
on bline_avg.sku_root_id = trans.sku_root_id
and bline_avg.store_id = trans.store_id
left join `gum-eroski-dev.promo_analysis.daily_sku_store_sale_average_dayweek` bline_avg_day
on bline_avg_day.sku_root_id = trans.sku_root_id
and bline_avg_day.store_id = trans.store_id
and bline_avg_day.dayweek = FORMAT_DATE('%A',trans.date) 

--Filter on sku, store cumulative distribution
CREATE OR REPLACE TABLE
`gum-eroski-dev.promo_analysis.sku_store_cum_dist` AS
WITH sku_store_bline_amt AS 
(SELECT distinct sku_root_id, store_id, bline_qty_all
FROM `gum-eroski-dev.promo_analysis.daily_total_aggregate_promo_to_sku_bline` 
), sku_store_cum_distribution AS (
SELECT 
*,
CUME_DIST() OVER ( PARTITION BY sku_root_id
        ORDER BY bline_qty_all ASC
) as cum_dist
FROM sku_store_bline_amt
) SELECT *
FROM sku_store_cum_distribution;

-- Get filtered cumulative distribution table
CREATE OR REPLACE TABLE
`gum-eroski-dev.promo_analysis.sku_store_cum_dist_filtered` AS
SELECT * FROM 
`gum-eroski-dev.promo_analysis.sku_store_cum_dist`
WHERE cum_dist>=0.1;

--Update the daily transaction to sku bline table to filter on sku+store combs in the cum table
CREATE OR REPLACE TABLE
`gum-eroski-dev.promo_analysis.aggregate_daily_transaction_to_sku_bline_filtered`  
PARTITION BY 
date
CLUSTER BY
store_id,sku_root_id,area,section
AS
SELECT trans.*,
FROM `gum-eroski-dev.promo_analysis.aggregate_daily_transaction_to_sku_bline` trans
INNER join `gum-eroski-dev.promo_analysis.sku_store_cum_dist_filtered` cum_filtered
on cum_filtered.sku_root_id = trans.sku_root_id
and cum_filtered.store_id = trans.store_id;

--Update the aggregate daily promo table to filter on sku+store combs in the cum table and remove cases 
--where n_promo>1 and n_promo_mech>1
CREATE OR REPLACE TABLE
`gum-eroski-dev.promo_analysis.daily_total_aggregate_promo_to_sku_bline_filtered`  
PARTITION BY 
date
CLUSTER BY
store_id,sku_root_id,area,section
AS
SELECT promo.*,
FROM `gum-eroski-dev.promo_analysis.daily_total_aggregate_promo_to_sku_bline` promo
INNER join `gum-eroski-dev.promo_analysis.sku_store_cum_dist_filtered` cum_filtered
on cum_filtered.sku_root_id = promo.sku_root_id
and cum_filtered.store_id = promo.store_id
WHERE n_promo=1
and n_promo_mech=1;

--Update the daily aggregate promo to sku pf table to filter on sku+store combs in the cum table and remove cases 
--where n_promo>1 and n_promo_mech>1
CREATE OR REPLACE TABLE
`gum-eroski-dev.promo_analysis.daily_total_aggregate_promo_to_sku_pf_filtered`  
PARTITION BY 
date
CLUSTER BY
store_id,sku_root_id,area,section
AS
SELECT promo.*,
FROM `gum-eroski-dev.promo_analysis.daily_total_aggregate_promo_to_sku_pf` promo
INNER join `gum-eroski-dev.promo_analysis.sku_store_cum_dist_filtered` cum_filtered
on cum_filtered.sku_root_id = promo.sku_root_id
and cum_filtered.store_id = promo.store_id
WHERE n_promo=1
and n_promo_mech=1;

-- Get the daily sales for the year for the sku
SELECT
store_id,
area,
section,
category,
subcategory,
segment,
date,
SUM(total_sale_amt) as total_sale_amt,
SUM(total_sale_qty) as total_sale_qty,
SUM(total_margin_amt) as total_margin_amt,
MAX(promo_flag) as promo_flag,
SUM(bline_qty_all) as bline_qty_all,
SUM(bline_qty_all_day) as bline_qty_all_day,
(SUM(total_sale_qty) - SUM(bline_qty_all)) as residuals_bline_qty,
(SUM(total_sale_qty) - SUM(bline_qty_all_day)) as residuals_bline_qty_day
FROM `promo_analysis.aggregate_daily_transaction_to_sku_bline` 
WHERE segment = "INFANTIL TIPO DINOS"
AND date between "2018-01-01" and "2018-12-31"
AND store_id = "733"
group by 
store_id,
area,
section,
category,
subcategory,
segment,
date
order by date;


-- Get the aggregate uplift % (qty) for a sku, mechanic, gondola, leaflet (for promo period)
WITH day_promo AS 
(SELECT *,
CASE WHEN duration in (7) THEN 1
WHEN duration in (14) THEN 2
WHEN duration in (21) THEN 3
WHEN duration in (28) THEN 4
ELSE NULL 
END AS duration_week,
(total_sale_qty-bline_qty_all_day) as inc_sale_qty,
DATE_DIFF(date, start_date, DAY)+1 AS n_days_since_promo_start
FROM `gum-eroski-dev.promo_analysis.daily_total_aggregate_promo_to_sku_bline_filtered` 
)
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
wow_flag,
promo_mechanic,
promo_mechanic_description,
discount_depth,
discount_depth_rank,
sum(total_sale_qty) as sum_total_sale_qty,
sum(bline_qty_all_day) as sum_bline_qty_all_day,
sum(inc_sale_qty) as sum_inc_sale_qty,
SAFE_DIVIDE(sum(inc_sale_qty), sum(bline_qty_all_day)) as avg_perc_inc_sale_qty,
n_days_since_promo_start,
duration_week,
in_gondola_flag,
in_leaflet_flag
FROM day_promo
WHERE 
sku_root_id = "11009842" --to change to a sku id of your choice
and duration_week in (1,2,3,4)
group by 
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
wow_flag,
promo_mechanic,
promo_mechanic_description,
discount_depth,
discount_depth_rank,
n_days_since_promo_start,
duration_week,
in_gondola_flag,
in_leaflet_flag
order by 
in_gondola_flag,
in_leaflet_flag,
duration_week, 
n_days_since_promo_start;


-- input data for post promo period

-- Get the aggregate input data buy sku, mechanic, gondola
WITH day_promo AS 
(SELECT *,
CASE WHEN duration in (7) THEN 1
WHEN duration in (14) THEN 2
WHEN duration in (21) THEN 3
WHEN duration in (28) THEN 4
ELSE NULL 
END AS duration_week,
(pf_sale_qty-bline_qty_all_day) as inc_pf_sale_qty,
SAFE_DIVIDE((pf_sale_qty-bline_qty_all_day),bline_qty_all_day) as perc_inc_pf_sale_qty,
DATE_DIFF(date, end_date, DAY)+1 AS n_days_since_promo_end
FROM `gum-eroski-dev.promo_analysis.daily_total_aggregate_promo_to_sku_pf_filtered` 
)
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
wow_flag,
promo_mechanic,
promo_mechanic_description,
discount_depth,
discount_depth_rank,
sum(pf_sale_qty) as sum_pf_sale_qty,
sum(bline_qty_all_day) as sum_bline_qty_all_day,
sum(inc_pf_sale_qty) as sum_inc_pf_sale_qty,
avg(pf_sale_qty) as avg_pf_sale_qty,
avg(bline_qty_all_day) as avg_bline_qty_all_day,
avg(inc_pf_sale_qty) as avg_inc_pf_sale_qty,
avg(perc_inc_pf_sale_qty) as avg_perc_inc_sale_qty,
n_days_since_promo_end,
duration_week,
in_gondola_flag
FROM day_promo
WHERE 
sku_root_id = "186288"
and duration_week in (1,2,3,4)
group by 
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
wow_flag,
promo_mechanic,
promo_mechanic_description,
discount_depth,
discount_depth_rank,
n_days_since_promo_end,
duration_week,
in_gondola_flag
order by 
in_gondola_flag,
duration_week, 
n_days_since_promo_end;;


/* -- Filter criteria SQL 
SELECT sum(total_sale_amt) as total_sale_amt, 
sum(total_sale_qty) as total_sale_qty
FROM 
(
SELECT period_active.*, 
trans_sale_amt.total_sale_amt,
trans_sale_amt.total_sale_qty   
FROM `gum-eroski-dev.promo_analysis.sku_store_div_period` period_active
LEFT join 
(SELECT sku_root_id, 
store_id, 
SUM(total_sale_amt) as total_sale_amt, 
SUM(total_sale_qty) as total_sale_qty
FROM `gum-eroski-dev.ETL.aggregate_daily_transaction_to_sku`
group by 
sku_root_id, 
store_id
) trans_sale_amt
ON trans_sale_amt.sku_root_id = period_active.sku_root_id
AND trans_sale_amt.store_id = period_active.store_id
where period_active.period>700
) t1;; */

---
WITH 
day_promo AS 
(SELECT *,
CASE WHEN duration in (7) THEN 1
WHEN duration in (14) THEN 2
WHEN duration in (21) THEN 3
WHEN duration in (28) THEN 4
ELSE NULL 
END AS duration_week
FROM `gum-eroski-dev.promo_analysis.daily_total_aggregate_promo_to_sku_pf_filtered` 
), 

store_exclusion AS (
SELECT distinct sku_root_id, store_id, CONCAT(sku_root_id, "_", store_id) as c_sku_store
FROM `gum-eroski-dev.promo_analysis.daily_total_aggregate_promo_to_sku_pf_filtered` 
where pf_promo_flag = 1
), 

day_filtered AS (
SELECT 
*,
(pf_sale_qty-bline_qty_all_day) as inc_pf_sale_qty,
SAFE_DIVIDE((pf_sale_qty-bline_qty_all_day),bline_qty_all_day) as perc_inc_pf_sale_qty,
DATE_DIFF(date, end_date, DAY)+1 AS n_days_since_promo_end
FROM day_promo
WHERE CONCAT(day_promo.sku_root_id, "_", day_promo.store_id) not in (select c_sku_store from store_exclusion)
)
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
wow_flag,
promo_mechanic,
promo_mechanic_description,
discount_depth,
discount_depth_rank,
sum(pf_sale_qty) as sum_pf_sale_qty,
sum(bline_qty_all_day) as sum_bline_qty_all_day,
sum(inc_pf_sale_qty) as sum_inc_pf_sale_qty,
SAFE_DIVIDE(sum(inc_pf_sale_qty),sum(bline_qty_all_day)) as perc_inc_sale_qty,
avg(pf_sale_qty) as avg_pf_sale_qty,
avg(bline_qty_all_day) as avg_bline_qty_all_day,
avg(inc_pf_sale_qty) as avg_inc_pf_sale_qty,
avg(perc_inc_pf_sale_qty) as avg_perc_inc_sale_qty,
n_days_since_promo_end,
duration_week,
in_gondola_flag
FROM day_filtered
WHERE 
sku_root_id = "186288"
and discount_depth = "buy 2 pay 1.5"
and duration_week in (1,2,3,4)
group by 
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
wow_flag,
promo_mechanic,
promo_mechanic_description,
discount_depth,
discount_depth_rank,
n_days_since_promo_end,
duration_week,
in_gondola_flag
order by 
in_gondola_flag,
duration_week, 
n_days_since_promo_end;

--aggregate promo to sku input data into model
CREATE OR REPLACE TABLE 
`gum-eroski-dev.promo_analysis.aggregate_promo_to_sku` 
AS
SELECT 
promo_id, 
promo_year, 
name, 
type, 
start_date,
end_date,
CASE WHEN ROUND(SAFE_DIVIDE(duration,7),0)<1
THEN 1
ELSE ROUND(SAFE_DIVIDE(duration,7),0) 
END AS duration_wks,
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
wow_flag,
promo_mechanic,
promo_mechanic_description,
discount_depth,
discount_depth_rank,
in_leaflet_flag,
in_gondola_flag,	
sum(total_sale_amt) as total_sale_amt,
sum(total_sale_qty) as total_sale_qty,
sum(total_margin_amt) as total_margin_amt,
sum(bline_qty_all_day) as total_baseline_qty,
sum(total_sale_qty)-sum(bline_qty_all_day) as total_inc_qty,
SAFE_DIVIDE(sum(total_sale_qty)-sum(bline_qty_all_day),sum(bline_qty_all_day)) as perc_inc_qty,
sum(total_discount_amt) as total_discount_amt,
sum(oferta_promo_total_sale_amt) as oferta_promo_total_sale_amt,
sum(oferta_promo_total_sale_qty) as oferta_promo_total_sale_qty,
sum(oferta_promo_total_margin_amt) as oferta_promo_total_margin_amt,
sum(oferta_promo_total_discount_amt) as oferta_promo_total_discount_amt,
max(promo_prev_week) as promo_prev_week,
max(promo_pf_week) as promo_pf_week

FROM `gum-eroski-dev.promo_analysis.daily_total_aggregate_promo_to_sku_bline_filtered` 

group by 

promo_id, 
promo_year, 
name, 
type, 
start_date,
end_date,
duration_wks,
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
wow_flag,
promo_mechanic,
promo_mechanic_description,
discount_depth,
discount_depth_rank,
in_leaflet_flag,
in_gondola_flag;


-- For model train on example categories
SELECT 
*
FROM `gum-eroski-dev.promo_analysis.aggregate_promo_to_sku` 
where area in ("ALIMENTACION", "FRESCOS")
and section in ("LACTEOS", "DULCE", "DROGUERIA", "CARNICERIA")
and category in ("LECHE", "GALLETAS", "DETERGENCIA ROPA", "PAVO Y OTRAS AVES ENVASADO")
and promo_mechanic in ("10", "20")
and discount_depth is not null;








