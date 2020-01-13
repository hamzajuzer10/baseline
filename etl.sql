  ##Create an ETL dataset (to do)
  ##Create date and week start parameters (to do)
  ##Create a calendar table
CREATE OR REPLACE TABLE
  `ETL.calendar` AS
SELECT
  day AS date,
  CASE EXTRACT(DAYOFWEEK
  FROM
    day)
    WHEN 1 THEN 'Sunday'
    WHEN 2 THEN 'Monday'
    WHEN 3 THEN 'Tuesday'
    WHEN 4 THEN 'Wednesday'
    WHEN 5 THEN 'Thursday'
    WHEN 6 THEN 'Friday'
    WHEN 7 THEN 'Saturday'
END
  AS day,
  #extracts week that begins on Monday
  EXTRACT(ISOWEEK
  FROM
    day) AS week,
  CONCAT(CAST(EXTRACT(YEAR
      FROM
        day) AS STRING),'SEM', FORMAT("%02d",(EXTRACT(ISOWEEK
        FROM
          day)))) AS year_sem,
  CASE EXTRACT(MONTH
  FROM
    day)
    WHEN 1 THEN 'Jan'
    WHEN 2 THEN 'Feb'
    WHEN 3 THEN 'Mar'
    WHEN 4 THEN 'Apr'
    WHEN 5 THEN 'May'
    WHEN 6 THEN 'Jun'
    WHEN 7 THEN 'Jul'
    WHEN 8 THEN 'Aug'
    WHEN 9 THEN 'Sept'
    WHEN 10 THEN 'Oct'
    WHEN 11 THEN 'Nov'
    WHEN 12 THEN 'Dec'
END
  AS month,
  CASE EXTRACT(DAYOFWEEK
  FROM
    day)
    WHEN 1 THEN TRUE
    WHEN 2 THEN FALSE
    WHEN 3 THEN FALSE
    WHEN 4 THEN FALSE
    WHEN 5 THEN FALSE
    WHEN 6 THEN FALSE
    WHEN 7 THEN TRUE
END
  AS weekend,
  #is it a weekend
  EXTRACT(YEAR
  FROM
    day) AS year,
  EXTRACT(QUARTER
  FROM
    day) AS quarter,
  NULL AS national_holiday_1,
  NULL AS national_holiday_2,
  NULL AS national_holiday_3,
  NULL AS national_event_1,
  NULL AS national_event_2,
  NULL AS national_event_3
FROM (
  SELECT
    day
  FROM
    UNNEST( GENERATE_DATE_ARRAY(DATE('2017-10-02'), DATE('2019-10-06'), INTERVAL 1 DAY) ) AS day )
ORDER BY
  date ASC;
  ##Create a sku table
CREATE OR REPLACE TABLE
  `ETL.sku` AS
WITH
  sales_loc_t1 AS (
  SELECT
    CAST(COD_ARTICULO AS STRING) AS sku_id,
    COD_AMBITO AS sales_loc_code,
    AMBITO AS sales_loc
  FROM
    `gum-eroski-dev.source_data.24_M_ART_CC`
  WHERE
    COD_NIVEL_ESTR_LOC = 30632
  GROUP BY
    COD_ARTICULO,
    COD_AMBITO,
    AMBITO),
  sales_loc_t2 AS (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY sku_id ORDER BY sales_loc_code ASC) AS row_n
  FROM
    sales_loc_t1),
  sales_loc AS (
  SELECT
    * EXCEPT(row_n)
  FROM
    sales_loc_t2
  WHERE
    row_n = 1 ),
  num_f AS (
  SELECT
    *
  FROM
    `gum-eroski-dev.source_data.28_NETO_COMERCIAL_REF`
  WHERE
    REGEXP_CONTAINS(NETO_COMERCIAL, r'^[0-9\,]+$')),
  sku_cost AS (
  SELECT
    PROVR_GEN,
    PROVR_TRABAJO,
    NOMBRE,
    COD_ARTICULO,
    CAST(REPLACE(NETO_COMERCIAL, ',','.') AS NUMERIC) AS NETO_COMERCIAL
  FROM
    num_f),
  table AS (
  SELECT
    *,
    CONCAT(COD_N1_PPAL,"-",COD_N2_PPAL,"-",COD_N3_PPAL,"-",COD_N4_PPAL,"-",COD_N5_PPAL ) AS l5
  FROM
    `source_data.11_M_ARTICULOS`),
  table2 AS (
  SELECT
    *,
  IF
    (DESC_N1 IS NOT NULL,
      DESC_N1,
      area2) AS area,
  IF
    (DESC_N2 IS NOT NULL,
      DESC_N2,
      section2) AS section,
  IF
    (DESC_N3 IS NOT NULL,
      DESC_N3,
      category2) AS category,
  IF
    (DESC_N4 IS NOT NULL,
      DESC_N4,
      subcategory2) AS subcategory,
  IF
    (DESC_N5 IS NOT NULL,
      DESC_N5,
      segment2) AS segment
  FROM
    table
  LEFT JOIN
    `source_data.product_structure`
  USING
    (l5))
SELECT
  t11.COD_ART AS sku_id,
  t11.COD_ART_PRIM AS sku_root_id,
  t11.COD_EAN AS sku_ean_id,
  t11.DESC_TP_ARTICULO AS type,
  t11.DESC_ART AS description,
  t11.COD_N1_PPAL AS area_id,
  t11.area,
  t11.COD_N2_PPAL AS section_id,
  t11.section,
  t11.COD_N3_PPAL AS category_id,
  t11.category,
  t11.COD_N4_PPAL AS subcategory_id,
  t11.subcategory,
  t11.COD_N5_PPAL AS segment_id,
  t11.segment,
  CASE
    WHEN erk.FLG_MMPP = 1 THEN 'EROSKI'
  ELSE
  t11.COD_MARCA
END
  AS brand_name,
  t11.COD_TALLA AS size,
  t11.DESC_TALLA AS size_desc,
  t11.COD_TIPO_FORMATO AS unit_desc,
  t11.COD_FORMATO AS unit,
  t11.COD_PACK AS units_per_pack,
  DESC_COLOR AS color,
  CASE
    WHEN erk.FLG_MMPP = 1 THEN erk.Premium_Normal_Non_premium
  ELSE
  NULL
END
  AS eroskibrand_label,
  CASE
    WHEN erk.FLG_MMPP = 1 THEN 'Y'
  ELSE
  'N'
END
  AS eroskibrand_flag,
  t11.FLG_SALUD_BIENESTAR AS flag_healthy,
  t2.MARCADOR_CASTE AS wow_flag,
  t11.FLG_INNOVACION AS innovation_flag,
  t11.FLG_GAMA_TURISTICA AS tourism_flag,
  t11.FLG_GAMA_LOCAL AS local_flag,
  t11.FLG_GAMA_REGIONAL AS regional_flag,
  t11.FLG_PODER_ADQUISITIVO AS wealthy_range_flag,
  t11.COD_PROVR_GEN AS generic_vendor_code,
  t11.COD_PROVR_TRABAJO AS working_vendor_code,
  t11.NOMBRE AS vendor_name,
  cost.NETO_COMERCIAL AS cost_per_unit,
  sales_loc.sales_loc,
  sales_loc.sales_loc_code
FROM
  table2 t11
LEFT JOIN
  `source_data.2_M_ART_MARCADORES_` t2
ON
  t11.COD_ART = t2.COD_ARTICULO
LEFT JOIN
  `source_data.eroski_brand` erk
ON
  erk.COD_MARCA = t11.COD_MARCA
  AND erk.DESC_TIPOMARCA = t11.DESC_TIPOMARCA
  AND erk.DESC_TIPOMARCA2 = t11.DESC_TIPOMARCA2
  AND CAST(erk.COD_POSICION_MARCA AS STRING) = t11.COD_POSICION_MARCA
  AND erk.DESC_POSICION_MARCA = t11.DESC_POSICION_MARCA
LEFT JOIN
  sku_cost cost
ON
  cost.PROVR_GEN = t11.COD_PROVR_GEN
  AND cost.PROVR_TRABAJO = t11.COD_PROVR_TRABAJO
  AND cost.NOMBRE = t11.NOMBRE
  AND cost.COD_ARTICULO = t11.COD_ART
LEFT JOIN
  sales_loc sales_loc
ON
  sales_loc.sku_id = t11.COD_ART
WHERE
  t11.FLG_BLOQ_DEFINITIVO = 'N'
  AND t11.COD_ART IS NOT NULL
  AND t11.COD_ART_PRIM IS NOT NULL;
  #create store table
CREATE OR REPLACE TABLE
  `ETL.store` AS
SELECT
  COD_LOC AS store_id,
  DESC_LOC AS name,
  DESC_PROVIN AS store_geography_region,
  DESC_POBLACION AS store_geography_location,
  COD_POSTAL AS store_geography_postal,
  DESC_DIRECCION AS store_geography_address,
  DESC_SOCIEDAD AS business_area,
  DESC_NEGOCIO AS business_desc,
  DES_N1 AS store_type1,
  DESC_N2 AS store_type2,
  DESC_TP_LOC store_type_location,
  FLG_CUOTA AS market_share,
  FEC_INI_LOC AS opening_date,
  FEC_FIN_LOC AS closing_date,
  NUM_M2 AS size_m2,
  NUM_CAJAS AS n_isles,
  FLG_LEAN AS store_desc_1,
  FLG_TRANSFORMADO AS store_desc_2,
  FLG_PUESTA_PUNTO_PLUS AS store_desc_3
FROM
  `source_data.16_LOCALIZATIONES`
WHERE
  DESC_SOCIEDAD IN ("EROSKI S.COOP.",
    "CECOSA HIPERMERCADOS, S.L.",
    "EQUIPAMIENTO FAMILIAR Y SERVIC",
    "CECOSA SUPERMERCADOS, S.L.U.");
  #create promotional_campaign table
CREATE OR REPLACE TABLE
  `ETL.promotional_campaign`
PARTITION BY
  RANGE_BUCKET(promo_year,
    GENERATE_ARRAY(2017, 2020, 1))
CLUSTER BY
  promo_id,
  type,
  name,
  class AS
SELECT
  COD_PROMOCION AS promo_id,
  CAST(EJER_PROMOCION AS INT64) AS promo_year,
  DESCRIPCION AS name,
  COD_TP_PROMOCION AS type,
  PARSE_DATE ("%Y%m%d",
    FEC_INICIO) AS start_date,
  PARSE_DATE ("%Y%m%d",
    FEC_FIN) AS end_date,
  COD_TP_IMP_OFERTA AS class,
  COD_PERFIL AS customer_profile_type,
  DESC_PLAN_PUBLI AS marketing_type
FROM
  `source_data.17_M_OFER_PROMO_`
INNER JOIN
  `ETL.calendar` cal_start
ON
  cal_start.date = PARSE_DATE ("%Y%m%d",
    FEC_INICIO)
INNER JOIN
  `ETL.calendar` cal_end
ON
  cal_end.date = PARSE_DATE ("%Y%m%d",
    FEC_FIN)
WHERE
  PARSE_DATE ("%Y%m%d",
    FEC_FIN) >= PARSE_DATE ("%Y%m%d",
    FEC_INICIO);
  #create promotional_to_sku table
CREATE OR REPLACE TABLE
  `ETL.promotional_to_sku`
PARTITION BY
  RANGE_BUCKET(promo_year,
    GENERATE_ARRAY(2017, 2020, 1))
CLUSTER BY
  promo_id,
  sku_id,
  sku_root_id,
  store_id AS
WITH
  oferta_promo_to_sku AS (
  SELECT
    oferta.year_sem,
    oferta.COD_OFERTA AS promo_id,
    CAST(oferta.ANO_OFERTA AS INT64) AS promo_year,
    oferta.COD_ART AS sku_id,
    sku.sku_root_id AS sku_root_id,
    oferta.COD_LOC AS store_id,
    oferta.COD_TIPOOFERTA AS promo_mechanic,
    CASE
      WHEN oferta.COD_TIPOOFERTA='10' THEN 'OFFER PRICE-PRODUCT'
      WHEN oferta.COD_TIPOOFERTA='20' THEN 'OFFER N * M CONSTANT'
      WHEN oferta.COD_TIPOOFERTA='30' THEN 'OFFER N * H heterogenous SAME PVP'
      WHEN oferta.COD_TIPOOFERTA='40' THEN 'OFFER N * H heterogenous GIFT'
      WHEN oferta.COD_TIPOOFERTA='50' THEN 'GIFT OFFER'
      WHEN oferta.COD_TIPOOFERTA='60' THEN 'N OFFER BETTER THAN M'
      WHEN oferta.COD_TIPOOFERTA='70' THEN 'Virtual LOT'
      WHEN oferta.COD_TIPOOFERTA='72' THEN 'LOT VIRTUAL GIFT'
      WHEN oferta.COD_TIPOOFERTA='73' THEN 'VIRTUAL LOT BLOCK UNICO'
      WHEN oferta.COD_TIPOOFERTA='75' THEN 'Savings offer'
      WHEN oferta.COD_TIPOOFERTA='99' THEN 'FICTIONAL'
    ELSE
    NULL
  END
    AS promo_mechanic_description,
    CASE
      WHEN (oferta.COD_TIPOOFERTA IN ('60') AND oferta.UMBRAL1 IN (0) AND oferta.PVPUMBRAL1 IS NOT NULL) THEN oferta.PVPUMBRAL1
    ELSE
    oferta.PVP_TARIFA
  END
    AS std_price,
	CASE
      WHEN oferta.COD_TIPOOFERTA IN ('10') THEN CAST(oferta.COD_PRECIO AS NUMERIC)
    ELSE
    NULL
  END AS discounted_price_promo,
   CASE
      WHEN oferta.COD_TIPOOFERTA IN ('20', '30', '40') THEN 100*(1-(SAFE_DIVIDE(oferta.CANT_M,oferta.CANT_N)))
	  WHEN (oferta.COD_TIPOOFERTA IN ('60') 
	  AND oferta.UMBRAL1 IN (1) 
	  AND oferta.PVPUMBRAL1 IS NULL 
	  AND oferta.DESCUMBRAL1 IS NOT NULL 
	  AND oferta.UMBRAL2 IS NULL AND oferta.DESCUMBRAL1>5) THEN CAST(ROUND(oferta.DESCUMBRAL1/5,0)*5 AS NUMERIC)
      WHEN (oferta.COD_TIPOOFERTA IN ('60')
      AND oferta.UMBRAL1 IN (1)
      AND oferta.PVPUMBRAL1 IS NULL
      AND oferta.DESCUMBRAL1 IS NOT NULL
      AND oferta.UMBRAL2 IS NULL
      AND oferta.DESCUMBRAL1<=5
      AND oferta.DESCUMBRAL1>=0) THEN CAST(ROUND(oferta.DESCUMBRAL1/2.5,0)*2.5 AS NUMERIC)
      WHEN (oferta.COD_TIPOOFERTA IN ('60') 
	  AND oferta.UMBRAL1 IN (0) 
	  AND oferta.PVPUMBRAL1 IS NOT NULL 
	  AND oferta.UMBRAL2 IN (1) 
	  AND oferta.PVPUMBRAL2 IS NULL 
	  AND oferta.DESCUMBRAL2 IS NOT NULL 
	  AND oferta.UMBRAL3 IS NULL AND oferta.DESCUMBRAL2>5) THEN CAST(ROUND(oferta.DESCUMBRAL2/5,0)*5 AS NUMERIC)
      WHEN (oferta.COD_TIPOOFERTA IN ('60')
      AND oferta.UMBRAL1 IN (0)
      AND oferta.PVPUMBRAL1 IS NOT NULL
      AND oferta.UMBRAL2 IN (1)
      AND oferta.PVPUMBRAL2 IS NULL
      AND oferta.DESCUMBRAL2 IS NOT NULL
      AND oferta.UMBRAL3 IS NULL
      AND oferta.DESCUMBRAL2<=5
      AND oferta.DESCUMBRAL2>=0) THEN CAST(ROUND(oferta.DESCUMBRAL2/2.5,0)*2.5 AS NUMERIC)
      WHEN (oferta.COD_TIPOOFERTA IN ('60') 
	  AND oferta.UMBRAL1 IN (0) 
	  AND oferta.PVPUMBRAL1 IS NOT NULL 
	  AND oferta.UMBRAL2 IS NOT NULL 
	  AND oferta.PVPUMBRAL2 IS NOT NULL 
	  AND oferta.DESCUMBRAL2 IS NULL 
	  AND oferta.UMBRAL3 IS NULL 
	  AND (100*(1-SAFE_DIVIDE(oferta.PVPUMBRAL2, oferta.PVPUMBRAL1)))>5) THEN CAST(ROUND((100*(1-SAFE_DIVIDE(oferta.PVPUMBRAL2, oferta.PVPUMBRAL1)))/5,0)*5 AS NUMERIC)
      WHEN (oferta.COD_TIPOOFERTA IN ('60')
      AND oferta.UMBRAL1 IN (0)
      AND oferta.PVPUMBRAL1 IS NOT NULL
      AND oferta.UMBRAL2 IS NOT NULL
      AND oferta.PVPUMBRAL2 IS NOT NULL
      AND oferta.DESCUMBRAL2 IS NULL
      AND oferta.UMBRAL3 IS NULL
      AND (100*(1-SAFE_DIVIDE(oferta.PVPUMBRAL2,
            oferta.PVPUMBRAL1)))<=5
      AND (100*(1-SAFE_DIVIDE(oferta.PVPUMBRAL2,
            oferta.PVPUMBRAL1)))>=0) THEN CAST(ROUND((100*(1-SAFE_DIVIDE(oferta.PVPUMBRAL2,
                oferta.PVPUMBRAL1)))/2.5,0)*2.5 AS NUMERIC)
    ELSE
    NULL
  END AS discount_depth_rank,
    oferta.CANT_M AS no_to_pay,
    oferta.CANT_N AS no_to_buy,
    CASE
      WHEN oferta.COD_TIPOOFERTA IN ('20', '30', '40') THEN CONCAT("buy ", CAST(oferta.CANT_N AS STRING), " pay ", CAST(oferta.CANT_M AS STRING))
      WHEN (oferta.COD_TIPOOFERTA IN ('60')
      AND oferta.UMBRAL1 IN (1)
      AND oferta.PVPUMBRAL1 IS NOT NULL
      AND oferta.UMBRAL2 IS NULL) THEN CONCAT("buy 1 at ", CAST(oferta.PVPUMBRAL1 AS STRING))
      WHEN (oferta.COD_TIPOOFERTA IN ('60') AND oferta.UMBRAL1 IN (1) AND oferta.PVPUMBRAL1 IS NOT NULL AND oferta.UMBRAL2 IS NOT NULL AND oferta.PVPUMBRAL2 IS NOT NULL AND oferta.UMBRAL3 IS NULL) THEN CONCAT("buy 1 at ", CAST(oferta.PVPUMBRAL1 AS STRING), ", buy ", CAST(oferta.UMBRAL2 AS STRING)," at ",CAST(oferta.PVPUMBRAL2 AS STRING))
      WHEN (oferta.COD_TIPOOFERTA IN ('60')
      AND oferta.UMBRAL1 IN (1)
      AND oferta.PVPUMBRAL1 IS NOT NULL
      AND oferta.UMBRAL2 IS NOT NULL
      AND oferta.PVPUMBRAL2 IS NOT NULL
      AND oferta.UMBRAL3 IS NOT NULL
      AND oferta.PVPUMBRAL3 IS NOT NULL) THEN CONCAT("buy 1 at ", CAST(oferta.PVPUMBRAL1 AS STRING), ", buy ",CAST(oferta.UMBRAL2 AS STRING)," at ",CAST(oferta.PVPUMBRAL2 AS STRING), ", buy ",CAST(oferta.UMBRAL3 AS STRING)," at ",CAST(oferta.PVPUMBRAL3 AS STRING))
      WHEN (oferta.COD_TIPOOFERTA IN ('60') 
	  AND oferta.UMBRAL1 IN (1) 
	  AND oferta.PVPUMBRAL1 IS NULL 
	  AND oferta.DESCUMBRAL1 IS NOT NULL 
	  AND oferta.UMBRAL2 IS NULL AND oferta.DESCUMBRAL1>5) THEN CONCAT(CAST(ROUND(oferta.DESCUMBRAL1/5,0)*5 AS STRING), "% off")
      WHEN (oferta.COD_TIPOOFERTA IN ('60')
      AND oferta.UMBRAL1 IN (1)
      AND oferta.PVPUMBRAL1 IS NULL
      AND oferta.DESCUMBRAL1 IS NOT NULL
      AND oferta.UMBRAL2 IS NULL
      AND oferta.DESCUMBRAL1<=5
      AND oferta.DESCUMBRAL1>=0) THEN CONCAT(CAST(ROUND(oferta.DESCUMBRAL1/2.5,0)*2.5 AS STRING), "% off")
      WHEN (oferta.COD_TIPOOFERTA IN ('60') 
	  AND oferta.UMBRAL1 IN (0) 
	  AND oferta.PVPUMBRAL1 IS NOT NULL 
	  AND oferta.UMBRAL2 IN (1) 
	  AND oferta.PVPUMBRAL2 IS NULL 
	  AND oferta.DESCUMBRAL2 IS NOT NULL 
	  AND oferta.UMBRAL3 IS NULL AND oferta.DESCUMBRAL2>5) THEN CONCAT(CAST(ROUND(oferta.DESCUMBRAL2/5,0)*5 AS STRING), "% off")
      WHEN (oferta.COD_TIPOOFERTA IN ('60')
      AND oferta.UMBRAL1 IN (0)
      AND oferta.PVPUMBRAL1 IS NOT NULL
      AND oferta.UMBRAL2 IN (1)
      AND oferta.PVPUMBRAL2 IS NULL
      AND oferta.DESCUMBRAL2 IS NOT NULL
      AND oferta.UMBRAL3 IS NULL
      AND oferta.DESCUMBRAL2<=5
      AND oferta.DESCUMBRAL2>=0) THEN CONCAT(CAST(ROUND(oferta.DESCUMBRAL2/2.5,0)*2.5 AS STRING), "% off")
      WHEN (oferta.COD_TIPOOFERTA IN ('60') 
	  AND oferta.UMBRAL1 IN (0) 
	  AND oferta.PVPUMBRAL1 IS NOT NULL 
	  AND oferta.UMBRAL2 IS NOT NULL 
	  AND oferta.PVPUMBRAL2 IS NOT NULL 
	  AND oferta.DESCUMBRAL2 IS NULL 
	  AND oferta.UMBRAL3 IS NULL 
	  AND (100*(1-SAFE_DIVIDE(oferta.PVPUMBRAL2, oferta.PVPUMBRAL1)))>5) THEN CONCAT(CAST(ROUND((100*(1-SAFE_DIVIDE(oferta.PVPUMBRAL2, oferta.PVPUMBRAL1)))/5,0)*5 AS STRING),"% off")
      WHEN (oferta.COD_TIPOOFERTA IN ('60')
      AND oferta.UMBRAL1 IN (0)
      AND oferta.PVPUMBRAL1 IS NOT NULL
      AND oferta.UMBRAL2 IS NOT NULL
      AND oferta.PVPUMBRAL2 IS NOT NULL
      AND oferta.DESCUMBRAL2 IS NULL
      AND oferta.UMBRAL3 IS NULL
      AND (100*(1-SAFE_DIVIDE(oferta.PVPUMBRAL2,
            oferta.PVPUMBRAL1)))<=5
      AND (100*(1-SAFE_DIVIDE(oferta.PVPUMBRAL2,
            oferta.PVPUMBRAL1)))>=0) THEN CONCAT(CAST(ROUND((100*(1-SAFE_DIVIDE(oferta.PVPUMBRAL2,
                oferta.PVPUMBRAL1)))/2.5,0)*2.5 AS STRING),"% off")
    ELSE
    NULL
  END
    AS discount_depth,
    CASE
      WHEN oferta.COD_PORTADA IS NOT NULL THEN 1
    ELSE
    0
  END
    AS leaflet_cover,
    CASE
      WHEN oferta.COD_DESTACAR IS NOT NULL THEN 1
    ELSE
    0
  END
    AS leaflet_priv_space,
    CASE
      WHEN oferta.COD_FOLLETO IS NOT NULL THEN 1
    ELSE
    0
  END
    AS brochure,
    CASE
      WHEN oferta.COD_PORTADA IS NOT NULL OR oferta.COD_DESTACAR IS NOT NULL OR oferta.COD_FOLLETO IS NOT NULL THEN 1
    ELSE
    0
  END
    AS in_leaflet_flag,
    oferta.COD_GONDOLA AS gondola_desc,
    CASE
      WHEN oferta.COD_GONDOLA IS NOT NULL THEN 1
    ELSE
    0
  END
    AS in_gondola_flag,
    CASE
      WHEN ((oferta.COD_GONDOLA IS NOT NULL) AND (oferta.COD_PORTADA IS NOT NULL OR oferta.COD_DESTACAR IS NOT NULL OR oferta.COD_FOLLETO IS NOT NULL)) THEN 1
    ELSE
    0
  END
    AS in_both_leaflet_gondola_flag,
    oferta.COD_BLOQUE AS promo_group_code,
    oferta.IMPLANTAH AS marketing_ad_desc_1,
    oferta.IMPLANTAM AS marketing_ad_desc_2,
    oferta.IMPLANTAB AS marketing_ad_desc_3,
    oferta.CABSECCM AS store_placement_desc_1,
    oferta.CABSECCB AS store_placement_desc_2
  FROM
    `source_data.18_OFERTAS_ART_LOC` oferta
  INNER JOIN
    `ETL.promotional_campaign` promo
  ON
    oferta.COD_OFERTA = promo.promo_id
    AND CAST(oferta.ANO_OFERTA AS INT64) = promo.promo_year
  INNER JOIN
    `ETL.sku` sku
  ON
    oferta.COD_ART = sku.sku_id
  INNER JOIN
    `ETL.store` store
  ON
    oferta.COD_LOC = store.store_id )
SELECT
  DISTINCT *
FROM
  oferta_promo_to_sku;
   #create a transaction table with oferta descriptions
CREATE OR REPLACE TABLE
  `ETL.lin_transaction`
PARTITION BY
  date
CLUSTER BY
  sku_id,
  sku_root_id,
  store_id,
  sale_type AS
WITH
  promo_to_sku AS (
  SELECT
    DISTINCT promo_id,
    promo_year,
    sku_id,
    store_id
  FROM
    `gum-eroski-dev.ETL.promotional_to_sku` ),
  trans_prep AS (
  SELECT
    CONCAT(lin.COD_LOC, "_", lin.HORA_EMISION, "_", lin.COD_CAJA, "_", lin.NUM_TICKET, "_", lin.DIA) AS transaction_id,
    lin.COD_ART as sku_id,
    sku.sku_root_id AS sku_root_id,
    lin.COD_LOC as store_id,
    pr.promo_id AS promo_id,
    pr.promo_year AS promo_year,
    lin.ID_CLIENTE as customer_id,
    EXTRACT(DATE
    FROM
      PARSE_DATETIME ("%Y%m%d %H:%M",
        lin.HORA_EMISION)) AS date,
    EXTRACT(TIME
    FROM
      PARSE_DATETIME ("%Y%m%d %H:%M",
        lin.HORA_EMISION)) AS time,
    lin.IMP_PVP_TARIFA,
    lin.IMP_VENTA_TARIFA,
    lin.IMP_VENTA_OFERTA,
    lin.IMP_VENTA_LIQUID,
    lin.IMP_VENTA_CAMPANA,
    lin.IMP_VENTA_COMPETE,
    lin.IMP_VENTA_TARIFA + lin.IMP_VENTA_OFERTA + lin.IMP_VENTA_LIQUID + lin.IMP_VENTA_CAMPANA + lin.IMP_VENTA_COMPETE AS total_sale_amt,
    lin.UNID_VENTA_TARIFA,
    lin.UNID_VENTA_OFERTA,
    lin.UNID_VENTA_LIQUID,
    lin.UNID_VENTA_CAMPANA,
    lin.UNID_VENTA_COMPETE,
    lin.UNID_VENTA_TARIFA + lin.UNID_VENTA_OFERTA + lin.UNID_VENTA_LIQUID + lin.UNID_VENTA_CAMPANA + lin.UNID_VENTA_COMPETE AS total_sale_qty,
    lin.MARGEN_TARIFA,
    lin.MARGEN_OFERTA,
    lin.MARGEN_LIQUID,
    lin.MARGEN_CAMPANA,
    lin.MARGEN_COMPETE,
    lin.MARGEN_TARIFA + lin.MARGEN_OFERTA + lin.MARGEN_LIQUID + lin.MARGEN_CAMPANA + lin.MARGEN_COMPETE AS total_margin_amt,
    lin.IMP_DTO_VALE,
    lin.IMP_DTO_TRAVEL,
    lin.IMP_DTO_TMONEDERO,
    lin.IMP_DTO_OTROS,
    lin.IMP_DTO_ONSITE,
    lin.IMP_DTO_CUPON,
    lin.IMP_DTO_CUOTA,
    lin.IMP_DTO_CONSUMER,
    lin.IMP_CONSUMO_RAP,
    lin.IMP_DTO_VALE + lin.IMP_DTO_TRAVEL + lin.IMP_DTO_TMONEDERO + lin.IMP_DTO_OTROS + lin.IMP_DTO_ONSITE + lin.IMP_DTO_CUPON + lin.IMP_DTO_CUOTA +lin.IMP_DTO_CONSUMER +lin.IMP_CONSUMO_RAP AS total_non_oferta_discount_amt,
    lin.IMP_DTO_TRAVEL + lin.IMP_DTO_TMONEDERO + lin.IMP_DTO_OTROS + lin.IMP_DTO_ONSITE + lin.IMP_DTO_CUPON + lin.IMP_DTO_CUOTA +lin.IMP_DTO_CONSUMER +lin.IMP_CONSUMO_RAP AS total_non_oferta_discount_amt_erk
  FROM
    `source_data.7_ESTRUCTURA_LINEAS` lin
  INNER JOIN
    `gum-eroski-dev.ETL.store` store
  ON
    lin.COD_LOC = store.store_id
  INNER JOIN
    `ETL.calendar` cal
  ON
    cal.date = EXTRACT(DATE
    FROM
      PARSE_DATETIME ("%Y%m%d %H:%M",
        lin.HORA_EMISION))
  INNER JOIN
    `ETL.sku` sku
  ON
    sku.sku_id = lin.COD_ART
  LEFT JOIN
    promo_to_sku pr
  ON
    lin.COD_OFERTA = pr.promo_id
    AND CAST(lin.ANO_OFERTA AS INT64) = pr.promo_year
    AND lin.COD_ART = pr.sku_id
    AND lin.COD_LOC = pr.store_id
  WHERE
    lin.COD_TIPO_MOVIM = 'N'
    AND lin.COD_LOC IS NOT NULL
    AND lin.HORA_EMISION IS NOT NULL ),
  promo_trans_prep AS (
  SELECT
    * EXCEPT (promo_id),
    CASE promo_id
      WHEN '0' THEN NULL
    ELSE
    promo_id
  END
    AS promo_id
  FROM
    trans_prep )
SELECT
  *,
  CASE
    WHEN promo_id IS NOT NULL AND promo_year IS NOT NULL #This is an oferta promotion
  THEN CASE
    WHEN IMP_VENTA_OFERTA = IMP_PVP_TARIFA
  AND total_non_oferta_discount_amt=0 THEN 'Non-promo sale during oferta promo'
  ##to clarify with ERK - actual product price is higher then PVP Tarifa: this can happen sometimes due to non oferta discount
    WHEN IMP_VENTA_OFERTA = IMP_PVP_TARIFA AND total_non_oferta_discount_amt <> 0 THEN 'Non-oferta promo sale during oferta promo'
    WHEN IMP_PVP_TARIFA + total_non_oferta_discount_amt = IMP_VENTA_OFERTA THEN 'Non-oferta promo sale during oferta promo'
    WHEN IMP_PVP_TARIFA > IMP_VENTA_OFERTA AND total_non_oferta_discount_amt=0 THEN 'Oferta promo sale during oferta promo'
    WHEN IMP_PVP_TARIFA + total_non_oferta_discount_amt > IMP_VENTA_OFERTA
  AND total_non_oferta_discount_amt < 0 THEN 'Combo promo sale during oferta promo'
    WHEN IMP_VENTA_OFERTA > 0 THEN 'Oferta promo sale non matching during oferta promo' ##For all cases where the oferta sales dont match std price - discount
    WHEN (IMP_VENTA_TARIFA + IMP_VENTA_COMPETE + IMP_VENTA_LIQUID + IMP_VENTA_CAMPANA) > 0
  AND total_non_oferta_discount_amt <> 0
  AND IMP_PVP_TARIFA + total_non_oferta_discount_amt >= (IMP_VENTA_TARIFA + IMP_VENTA_COMPETE + IMP_VENTA_LIQUID + IMP_VENTA_CAMPANA) THEN 'Non-oferta promo sale during oferta promo'
  ELSE
  'Other sale during oferta promo'
END
  ELSE
  ##This is not an oferta promotion
  'Non-oferta sale'
END
  AS sale_type
FROM
  trans_prep;
  #create a transaction to sku table
CREATE OR REPLACE TABLE
  `ETL.transaction_to_sku`
PARTITION BY
  date
CLUSTER BY
  sku_id,
  sku_root_id,
  store_id,
  promo_id AS
WITH
  trans AS (
  SELECT
    transaction_id,
    sku_id,
    sku_root_id,
    store_id,
    promo_id,
    promo_year,
    customer_id,
    sale_type,
    date,
    time,
    IMP_VENTA_TARIFA AS total_sale_amt_std,
    IMP_VENTA_OFERTA AS total_sale_amt_discount,
    IMP_VENTA_LIQUID AS total_sale_amt_markdown,
    IMP_VENTA_CAMPANA AS total_sale_amt_campaign,
    IMP_VENTA_COMPETE AS total_sale_amt_other,
    total_sale_amt,
    UNID_VENTA_TARIFA AS total_sale_qty_std,
    UNID_VENTA_OFERTA AS total_sale_qty_discount,
    UNID_VENTA_LIQUID AS total_sale_qty_markdown,
    UNID_VENTA_CAMPANA AS total_sale_qty_campaign,
    UNID_VENTA_COMPETE AS total_sale_qty_other,
    total_sale_qty,
    MARGEN_TARIFA AS total_margin_amt_std,
    MARGEN_OFERTA AS total_margin_amt_discount,
    MARGEN_LIQUID AS total_margin_amt_markdown,
    MARGEN_CAMPANA AS total_margin_amt_campaign,
    MARGEN_COMPETE AS total_margin_amt_other,
    total_margin_amt,
    total_non_oferta_discount_amt,
    total_non_oferta_discount_amt_erk,
    IMP_PVP_TARIFA AS total_price_if_sku_std_price,
    SAFE_DIVIDE(IMP_PVP_TARIFA,
      total_sale_qty) AS std_price_per_unit,
    CASE
      WHEN sale_type = 'Oferta promo sale during oferta promo' THEN total_sale_amt
      WHEN sale_type IN ('Combo promo sale during oferta promo',
      'Oferta promo sale non matching during oferta promo') THEN IMP_VENTA_OFERTA
    ELSE
    0
  END
    AS oferta_promo_total_sale_amt,
    CASE
      WHEN sale_type = 'Oferta promo sale during oferta promo' THEN total_sale_qty
      WHEN sale_type IN ('Combo promo sale during oferta promo',
      'Oferta promo sale non matching during oferta promo') THEN UNID_VENTA_OFERTA
    ELSE
    0
  END
    AS oferta_promo_total_sale_qty,
    CASE
      WHEN sale_type = 'Oferta promo sale during oferta promo' THEN total_margin_amt
      WHEN sale_type IN ('Combo promo sale during oferta promo',
      'Oferta promo sale non matching during oferta promo') THEN MARGEN_OFERTA
    ELSE
    0
  END
    AS oferta_promo_total_margin_amt,
    CASE
      WHEN sale_type = 'Oferta promo sale during oferta promo' THEN -ABS(IMP_PVP_TARIFA-total_sale_amt)
      WHEN sale_type IN ('Combo promo sale during oferta promo',
      'Oferta promo sale non matching during oferta promo') THEN -ABS(IMP_PVP_TARIFA-total_sale_amt+total_non_oferta_discount_amt_erk)
    ELSE
    0
  END
    AS oferta_promo_total_discount_amt
  FROM
    ETL.lin_transaction)
SELECT
  transaction_id,
  sku_id,
  sku_root_id,
  store_id,
  promo_id,
  promo_year,
  customer_id,
  sale_type,
  date,
  time,
  SUM(total_sale_amt_std) AS total_sale_amt_std,
  SUM(total_sale_amt_discount) AS total_sale_amt_discount,
  SUM(total_sale_amt_markdown) AS total_sale_amt_markdown,
  SUM(total_sale_amt_campaign) AS total_sale_amt_campaign,
  SUM(total_sale_amt_other) AS total_sale_amt_other,
  SUM(total_sale_amt) AS total_sale_amt,
  SUM(total_sale_qty_std) AS total_sale_qty_std,
  SUM(total_sale_qty_discount) AS total_sale_qty_discount,
  SUM(total_sale_qty_markdown) AS total_sale_qty_markdown,
  SUM(total_sale_qty_campaign) AS total_sale_qty_campaign,
  SUM(total_sale_qty_other) AS total_sale_qty_other,
  SUM(total_sale_qty) AS total_sale_qty,
  SUM(total_margin_amt_std) AS total_margin_amt_std,
  SUM(total_margin_amt_discount) AS total_margin_amt_discount,
  SUM(total_margin_amt_markdown) AS total_margin_amt_markdown,
  SUM(total_margin_amt_campaign) AS total_margin_amt_campaign,
  SUM(total_margin_amt_other) AS total_margin_amt_other,
  SUM(total_margin_amt) AS total_margin_amt,
  SUM(total_non_oferta_discount_amt) AS total_non_oferta_discount_amt,
  SUM(total_non_oferta_discount_amt_erk) AS total_non_oferta_discount_amt_erk,
  SUM(total_price_if_sku_std_price) AS total_price_if_sku_std_price,
  AVG(std_price_per_unit) AS std_price_per_unit,
  SUM(oferta_promo_total_sale_amt) AS oferta_promo_total_sale_amt,
  SUM(oferta_promo_total_sale_qty) AS oferta_promo_total_sale_qty,
  SUM(oferta_promo_total_margin_amt) AS oferta_promo_total_margin_amt,
  SUM(oferta_promo_total_discount_amt) AS oferta_promo_total_discount_amt
FROM
  trans
GROUP BY
  transaction_id,
  sku_id,
  sku_root_id,
  store_id,
  promo_id,
  promo_year,
  customer_id,
  sale_type,
  date,
  time;
  #create transaction table
CREATE OR REPLACE TABLE
  `ETL.transaction`
PARTITION BY
  date
CLUSTER BY
  store_id,
  customer_id,
  type AS
WITH cab AS 
(
SELECT DISTINCT
CONCAT(cab.COD_LOC, "_", cab.HORA_EMISION, "_", cab.COD_CAJA, "_", cab.NUM_TICKET, "_", cab.DIA) as transaction_id,
cab.COD_TIPO_VENTA
FROM `gum-eroski-dev.source_data.6_E_CABECERAS` cab
)   
SELECT
  trans.transaction_id AS transaction_id,
  trans.store_id AS store_id,
  trans.date AS date,
  trans.time AS time,
  cab.COD_TIPO_VENTA AS type,
  trans.customer_id AS customer_id,
  SUM(trans.total_sale_amt) AS total_sale_amt,
  SUM(trans.total_sale_qty) AS total_sale_sku_qty,
  SUM(trans.total_margin_amt) AS total_margin,
  SUM(trans.total_non_oferta_discount_amt) AS total_non_oferta_discount_amt,
  SUM(trans.total_non_oferta_discount_amt_erk) AS total_non_oferta_discount_amt_erk,
  SUM(trans.total_price_if_sku_std_price) AS total_price_if_sku_std_price,
  SUM(trans.oferta_promo_total_sale_amt) AS oferta_promo_total_sale_amt,
  SUM(trans.oferta_promo_total_sale_qty) AS oferta_promo_total_sale_qty,
  SUM(trans.oferta_promo_total_margin_amt) AS oferta_promo_total_margin_amt,
  SUM(trans.oferta_promo_total_discount_amt) AS oferta_promo_total_discount_amt
FROM
  `gum-eroski-dev.ETL.transaction_to_sku` trans
LEFT JOIN
  cab
ON
  cab.transaction_id = trans.transaction_id
GROUP BY
  trans.transaction_id,
  trans.store_id,
  trans.date,
  trans.time,
  cab.COD_TIPO_VENTA,
  trans.customer_id;
  
## Create a temp promo to sku vendor funding table
CREATE OR REPLACE TABLE
  `ETL.temp_promo_to_sku_v_funding` 
  PARTITION BY
  date
  CLUSTER BY sku_id, sku_root_id, store_id
  OPTIONS( expiration_timestamp=TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY) ) 
  AS
WITH
  promo_campaign_cal AS (
  SELECT
    promo_id,
    promo_year,
    cal.date
  FROM
    `ETL.promotional_campaign`
  INNER JOIN
    `ETL.calendar` cal
  ON
    cal.date BETWEEN start_date
    AND end_date),
  promo_to_sku AS (
  SELECT
    promo.sku_id,
    promo.sku_root_id,
    promo.store_id,
    promo.promo_id,
    promo.promo_year

  FROM
    `ETL.promotional_to_sku` promo
  INNER JOIN
    `ETL.store` store
  ON
    store.store_id = promo.store_id
 
  GROUP BY
    promo.sku_id,
    promo.sku_root_id,
    promo.store_id,
    promo.promo_id,
    promo.promo_year
  )
  SELECT
    promo.*,
    promo_to_sku.* EXCEPT (promo_id,
      promo_year)
  FROM
    promo_to_sku promo_to_sku
  INNER JOIN
    promo_campaign_cal promo
  ON
    promo_to_sku.promo_id = promo.promo_id
    AND promo_to_sku.promo_year = promo.promo_year;

# create vendor funding table
CREATE OR REPLACE TABLE
  `ETL.vendor_funding` AS
WITH
  agg_trans AS (
  SELECT
    trans.sku_id,
    trans.sku_root_id,
    trans.date,
    trans.store_id,
    sum(trans.total_sale_qty) as total_sale_qty
  FROM
    `gum-eroski-dev.ETL.transaction_to_sku` trans
  GROUP BY
    trans.sku_id,
    trans.sku_root_id,
    trans.date,
    trans.store_id ),
   agg_v_funding AS ( 
    
    SELECT 
    PARSE_DATE("%Y%m%d",
      CAST(reden.DIA AS STRING)) as date,
    CAST(reden.COD_LOC AS STRING) as store_id,
    reden.COD_ART as sku_id,
    SUM(reden.UNID_VENTA) as total_expected_units,
    SUM(reden.IMP_REDENCION_PS+reden.IMP_REDENCION_OSS+reden.IMP_REDENCION) as total_vendor_funding_amt
    
    FROM `gum-eroski-dev.source_data.20_H_REDEN_TEOR_DIA` reden
    GROUP BY
    date,
    store_id,
    sku_id
  ),  
  v_funding AS (
  SELECT
    agg_trans.date,
    agg_trans.store_id,
    agg_trans.sku_id,
    agg_trans.sku_root_id,
    agg_trans.total_sale_qty,
    v_funding.total_expected_units,
    IFNULL(SAFE_DIVIDE(v_funding.total_vendor_funding_amt,agg_trans.total_sale_qty),0) AS expected_vendor_funding_per_unit,
    CASE WHEN temp_promo.promo_id is null
    THEN 0
    ELSE 1
    END AS promo_flag
  FROM
    agg_trans agg_trans
  LEFT JOIN
    agg_v_funding v_funding
  ON
    agg_trans.sku_id = v_funding.sku_id
    AND agg_trans.date = v_funding.date
    AND agg_trans.store_id = v_funding.store_id 
  LEFT JOIN `gum-eroski-dev.ETL.temp_promo_to_sku_v_funding` temp_promo
  ON
    agg_trans.sku_id = temp_promo.sku_id
    AND agg_trans.date = temp_promo.date
    AND agg_trans.store_id = temp_promo.store_id 
    )
SELECT
  sku_id,
  sku_root_id,
  AVG(expected_vendor_funding_per_unit) AS expected_vendor_funding_per_unit
FROM
  v_funding
WHERE promo_flag=1
GROUP BY
  sku_id,
  sku_root_id;
  
  #create root_sku table
CREATE OR REPLACE TABLE
  `ETL.root_sku` AS
WITH
  root_sku_sales AS (
  SELECT
    trans.sku_id,
    SUM(trans.total_sale_amt) AS total_sale_amt
  FROM
    `gum-eroski-dev.ETL.transaction_to_sku` trans
  GROUP BY
    trans.sku_id ),
  sku_with_vendor_funding AS (
	SELECT sku.*,
	vfunding.expected_vendor_funding_per_unit
	FROM `gum-eroski-dev.ETL.sku` sku
	LEFT JOIN `gum-eroski-dev.ETL.vendor_funding` vfunding
	on sku.sku_id = vfunding.sku_id
	),
  root_sku_sales_desc AS (
  SELECT
    sku.* EXCEPT(sku_id),
    sales.* EXCEPT(sku_id)
  FROM
    sku_with_vendor_funding sku
  INNER JOIN
    root_sku_sales sales
  ON
    sku.sku_id = sales.sku_id ),
  root_sku_max_sales_desc AS (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY sku_root_id ORDER BY total_sale_amt DESC) AS row_n
  FROM
    root_sku_sales_desc ),
  root_sku AS (
  SELECT
    * EXCEPT(total_sale_amt,
      row_n)
  FROM
    root_sku_max_sales_desc
  WHERE
    row_n = 1)
SELECT *
FROM root_sku;
# Create a temp aggregate promo to sku table
CREATE OR REPLACE TABLE
  `ETL.temp_aggregate_promo_to_sku`
PARTITION BY
  date
CLUSTER BY
  sku_root_id,
  store_id OPTIONS( expiration_timestamp=TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY) ) AS
WITH
  promo_campaign_cal AS (
  SELECT
    promo_id,
    promo_year,
    cal.date
  FROM
    `ETL.promotional_campaign`
  INNER JOIN
    `ETL.calendar` cal
  ON
    cal.date BETWEEN start_date
    AND end_date ),
  promo_to_sku AS (
  SELECT
    DISTINCT sku_root_id,
    store_id,
    promo_id,
    promo_year
  FROM
    `ETL.promotional_to_sku` )
SELECT
  promo.date,
  promo_to_sku.sku_root_id,
  promo_to_sku.store_id,
  promo_to_sku.promo_id,
  promo_to_sku.promo_year
FROM
  promo_to_sku
INNER JOIN
  promo_campaign_cal promo
ON
  promo_to_sku.promo_id = promo.promo_id
  AND promo_to_sku.promo_year = promo.promo_year
GROUP BY
  promo.date,
  promo_to_sku.sku_root_id,
  promo_to_sku.store_id,
  promo_to_sku.promo_id,
  promo_to_sku.promo_year;

#create a temp aggregate transaction to sku table
CREATE OR REPLACE TABLE
  `ETL.temp_aggregate_transaction_to_sku`
PARTITION BY
  date
CLUSTER BY
  sku_root_id,
  store_id,
  area,
  section OPTIONS( expiration_timestamp=TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY) ) AS
WITH promo_sku AS (
SELECT distinct 
date, 
sku_root_id,
store_id,
promo_id,
promo_year
FROM  `ETL.temp_aggregate_promo_to_sku`
)
SELECT
  trans.sku_root_id,
  trans.store_id,
  trans.date,
  promo_sku.promo_id,
  promo_sku.promo_year,
  root_sku.description,
  root_sku.area,
  root_sku.section,
  root_sku.category,
  root_sku.subcategory,
  root_sku.segment,
  root_sku.brand_name,
  root_sku.eroskibrand_flag,
  root_sku.eroskibrand_label,
  SUM(trans.total_price_if_sku_std_price) AS total_price_if_sku_std_price,
  AVG(trans.std_price_per_unit) AS std_price_per_unit,
  SUM(trans.total_sale_amt_std) AS total_sale_amt_std,
  SUM(trans.total_sale_amt_discount) AS total_sale_amt_discount,
  SUM(trans.total_sale_amt_markdown) AS total_sale_amt_markdown,
  SUM(trans.total_sale_amt_campaign) AS total_sale_amt_campaign,
  SUM(trans.total_sale_amt_other) AS total_sale_amt_other,
  SUM(trans.total_sale_amt) AS total_sale_amt,
  SUM(trans.total_sale_qty_std) AS total_sale_qty_std,
  SUM(trans.total_sale_qty_discount) AS total_sale_qty_discount,
  SUM(trans.total_sale_qty_markdown) AS total_sale_qty_markdown,
  SUM(trans.total_sale_qty_campaign) AS total_sale_qty_campaign,
  SUM(trans.total_sale_qty_other) AS total_sale_qty_other,
  SUM(trans.total_sale_qty) AS total_sale_qty,
  SUM(trans.total_margin_amt_std) AS total_margin_amt_std,
  SUM(trans.total_margin_amt_discount) AS total_margin_amt_discount,
  SUM(trans.total_margin_amt_markdown) AS total_margin_amt_markdown,
  SUM(trans.total_margin_amt_campaign) AS total_margin_amt_campaign,
  SUM(trans.total_margin_amt_other) AS total_margin_amt_other,
  SUM(trans.total_margin_amt) AS total_margin_amt,
  SUM(trans.total_non_oferta_discount_amt) AS total_non_oferta_discount_amt,
  SUM(trans.total_non_oferta_discount_amt_erk) AS total_non_oferta_discount_amt_erk,
  SUM(CASE WHEN promo_sku.promo_id is not NULL AND promo_sku.promo_year is not NULL then trans.oferta_promo_total_sale_amt
      ELSE 0 END) AS oferta_promo_total_sale_amt,
  SUM(CASE WHEN promo_sku.promo_id is not NULL AND promo_sku.promo_year is not NULL then trans.oferta_promo_total_sale_qty
      ELSE 0 END) AS oferta_promo_total_sale_qty,
  SUM(CASE WHEN promo_sku.promo_id is not NULL AND promo_sku.promo_year is not NULL then trans.oferta_promo_total_margin_amt
      ELSE 0 END) AS oferta_promo_total_margin_amt,
  SUM(CASE WHEN promo_sku.promo_id is not NULL AND promo_sku.promo_year is not NULL then trans.oferta_promo_total_discount_amt
      ELSE 0 END) AS oferta_promo_total_discount_amt
FROM
  `gum-eroski-dev.ETL.transaction_to_sku` trans
INNER JOIN
  `gum-eroski-dev.ETL.root_sku` root_sku
ON
  root_sku.sku_root_id = trans.sku_root_id
LEFT JOIN
  promo_sku
ON 
promo_sku.date = trans.date
AND promo_sku.sku_root_id = trans.sku_root_id
AND promo_sku.store_id = trans.store_id
AND promo_sku.promo_id = trans.promo_id
AND promo_sku.promo_year = trans.promo_year

GROUP BY
  trans.sku_root_id,
  trans.store_id,
  trans.date,
  promo_sku.promo_id,
  promo_sku.promo_year,
  root_sku.description,
  root_sku.area,
  root_sku.section,
  root_sku.category,
  root_sku.subcategory,
  root_sku.segment,
  root_sku.brand_name,
  root_sku.eroskibrand_flag,
  root_sku.eroskibrand_label;

 #create filled aggregate transaction to sku table
CREATE OR REPLACE TABLE
  `ETL.temp_filled_aggregate_transaction_to_sku`
PARTITION BY
  date
CLUSTER BY
  sku_root_id,
  store_id,
  area,
  section OPTIONS( expiration_timestamp=TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY) ) AS
WITH
  cal AS (
  SELECT
    date
  FROM
    `ETL.calendar`),
  sku_root_group AS (
  SELECT
    sku_root_id,
    store_id,
    description,
    area,
    section,
    category,
    subcategory,
    segment,
    brand_name,
    eroskibrand_flag,
    eroskibrand_label
  FROM
    `gum-eroski-dev.ETL.temp_aggregate_transaction_to_sku`
  GROUP BY
    sku_root_id,
    store_id,
    description,
    area,
    section,
    category,
    subcategory,
    segment,
    brand_name,
    eroskibrand_flag,
    eroskibrand_label)
  SELECT
    sku_root_id,
    store_id,
    description,
    area,
    section,
    category,
    subcategory,
    segment,
    brand_name,
    eroskibrand_flag,
    eroskibrand_label,
    date
  FROM
    sku_root_group
  CROSS JOIN
    cal;

# Create a filled aggregate transaction to sku + promo table
 CREATE OR REPLACE TABLE
  `ETL.temp_promo_filled_aggregate_transaction_to_sku`
PARTITION BY
  date
CLUSTER BY
  sku_root_id,
  store_id,
  area,
  section OPTIONS( expiration_timestamp=TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY) ) AS
SELECT trans.*, 
promo.promo_id,
promo.promo_year
FROM `gum-eroski-dev.ETL.temp_filled_aggregate_transaction_to_sku` trans
LEFT JOIN `ETL.temp_aggregate_promo_to_sku` promo
on trans.sku_root_id = promo.sku_root_id
and trans.store_id = promo.store_id
and trans.date = promo.date;

# Create a filled aggregate transaction to sku+ promo+ sales table
 CREATE OR REPLACE TABLE
  `ETL.temp_promo_sales_filled_aggregate_transaction_to_sku`
PARTITION BY
  date
CLUSTER BY
  sku_root_id,
  store_id,
  area,
  section OPTIONS( expiration_timestamp=TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY) ) AS
SELECT 
trans_filled.sku_root_id,
trans_filled.store_id,
trans_filled.description,
trans_filled.area,
trans_filled.section,
trans_filled.category,
trans_filled.subcategory,
trans_filled.segment,
trans_filled.brand_name,
trans_filled.eroskibrand_flag,
trans_filled.eroskibrand_label,
trans_filled.date,
trans_filled.promo_id,
trans_filled.promo_year,
IFNULL(trans.total_price_if_sku_std_price,0) as total_price_if_sku_std_price,
IFNULL(trans.std_price_per_unit,null) as std_price_per_unit,
IFNULL(trans.total_sale_amt_std,0) as total_sale_amt_std,
IFNULL(trans.total_sale_amt_discount,0) as total_sale_amt_discount,
IFNULL(trans.total_sale_amt_markdown,0) as total_sale_amt_markdown,
IFNULL(trans.total_sale_amt_campaign,0) as total_sale_amt_campaign,
IFNULL(trans.total_sale_amt_other,0) as total_sale_amt_other,
IFNULL(trans.total_sale_amt,0) as total_sale_amt,
IFNULL(trans.total_sale_qty_std,0) as total_sale_qty_std,
IFNULL(trans.total_sale_qty_discount,0) as total_sale_qty_discount,
IFNULL(trans.total_sale_qty_markdown,0) as total_sale_qty_markdown,
IFNULL(trans.total_sale_qty_campaign,0) as total_sale_qty_campaign,
IFNULL(trans.total_sale_qty_other,0) as total_sale_qty_other,
IFNULL(trans.total_sale_qty,0) as total_sale_qty,
IFNULL(trans.total_margin_amt_std,0) as total_margin_amt_std,
IFNULL(trans.total_margin_amt_discount,0) as total_margin_amt_discount,
IFNULL(trans.total_margin_amt_markdown,0) as total_margin_amt_markdown,
IFNULL(trans.total_margin_amt_campaign,0) as total_margin_amt_campaign,
IFNULL(trans.total_margin_amt_other,0) as total_margin_amt_other,
IFNULL(trans.total_margin_amt,0) as total_margin_amt,
IFNULL(trans.total_non_oferta_discount_amt,0) as total_non_oferta_discount_amt,
IFNULL(trans.total_non_oferta_discount_amt_erk,0) as total_non_oferta_discount_amt_erk,
IFNULL(trans.oferta_promo_total_sale_amt,0) as oferta_promo_total_sale_amt,
IFNULL(trans.oferta_promo_total_sale_qty,0) as oferta_promo_total_sale_qty,
IFNULL(trans.oferta_promo_total_margin_amt,0) as oferta_promo_total_margin_amt,
IFNULL(trans.oferta_promo_total_discount_amt,0) as oferta_promo_total_discount_amt

FROM `gum-eroski-dev.ETL.temp_promo_filled_aggregate_transaction_to_sku` trans_filled
LEFT JOIN `ETL.temp_aggregate_transaction_to_sku` trans
on trans.sku_root_id = trans_filled.sku_root_id
and trans.store_id = trans_filled.store_id
and trans.date = trans_filled.date
and trans.promo_id = trans_filled.promo_id
and trans.promo_year = trans_filled.promo_year

UNION DISTINCT

SELECT 
trans.sku_root_id,
trans.store_id,
trans.description,
trans.area,
trans.section,
trans.category,
trans.subcategory,
trans.segment,
trans.brand_name,
trans.eroskibrand_flag,
trans.eroskibrand_label,
trans.date,
trans.promo_id,
trans.promo_year,
trans.total_price_if_sku_std_price,
trans.std_price_per_unit,
trans.total_sale_amt_std,
trans.total_sale_amt_discount,
trans.total_sale_amt_markdown,
trans.total_sale_amt_campaign,
trans.total_sale_amt_other,
trans.total_sale_amt,
trans.total_sale_qty_std,
trans.total_sale_qty_discount,
trans.total_sale_qty_markdown,
trans.total_sale_qty_campaign,
trans.total_sale_qty_other,
trans.total_sale_qty,
trans.total_margin_amt_std,
trans.total_margin_amt_discount,
trans.total_margin_amt_markdown,
trans.total_margin_amt_campaign,
trans.total_margin_amt_other,
trans.total_margin_amt,
trans.total_non_oferta_discount_amt,
trans.total_non_oferta_discount_amt_erk,
trans.oferta_promo_total_sale_amt,
trans.oferta_promo_total_sale_qty,
trans.oferta_promo_total_margin_amt,
trans.oferta_promo_total_discount_amt

FROM `ETL.temp_aggregate_transaction_to_sku` trans;

  # Create aggregate daily transaction to sku promo
CREATE OR REPLACE TABLE
  `ETL.aggregate_daily_transaction_to_sku_promo`
PARTITION BY
  date
CLUSTER BY
  sku_root_id,
  store_id,
  category,
  promo_flag AS
WITH
  trans_to_sku AS (
  SELECT
    sku_root_id,
    store_id,
    description,
    area,
    section,
    category,
    subcategory,
    segment,
    brand_name,
    eroskibrand_flag,
    eroskibrand_label,
    date,
    promo_id,
    promo_year,
    SUM(total_price_if_sku_std_price) AS total_price_if_sku_std_price,
    AVG(std_price_per_unit) AS std_price_per_unit,
    SUM(total_sale_amt_std) AS total_sale_amt_std,
    SUM(total_sale_amt_discount) AS total_sale_amt_discount,
    SUM(total_sale_amt_markdown) AS total_sale_amt_markdown,
    SUM(total_sale_amt_campaign) AS total_sale_amt_campaign,
    SUM(total_sale_amt_other) AS total_sale_amt_other,
    SUM(total_sale_amt) AS total_sale_amt,
    SUM(total_sale_qty_std) AS total_sale_qty_std,
    SUM(total_sale_qty_discount) AS total_sale_qty_discount,
    SUM(total_sale_qty_markdown) AS total_sale_qty_markdown,
    SUM(total_sale_qty_campaign) AS total_sale_qty_campaign,
    SUM(total_sale_qty_other) AS total_sale_qty_other,
    SUM(total_sale_qty) AS total_sale_qty,
    SUM(total_margin_amt_std) AS total_margin_amt_std,
    SUM(total_margin_amt_discount) AS total_margin_amt_discount,
    SUM(total_margin_amt_markdown) AS total_margin_amt_markdown,
    SUM(total_margin_amt_campaign) AS total_margin_amt_campaign,
    SUM(total_margin_amt_other) AS total_margin_amt_other,
    SUM(total_margin_amt) AS total_margin_amt,
    SUM(total_non_oferta_discount_amt) AS total_non_oferta_discount_amt,
    SUM(total_non_oferta_discount_amt_erk) AS total_non_oferta_discount_amt_erk,
    SUM(oferta_promo_total_sale_amt) AS oferta_promo_total_sale_amt,
    SUM(oferta_promo_total_sale_qty) AS oferta_promo_total_sale_qty,
    SUM(oferta_promo_total_margin_amt) AS oferta_promo_total_margin_amt,
    SUM(oferta_promo_total_discount_amt) AS oferta_promo_total_discount_amt
  FROM
    `gum-eroski-dev.ETL.temp_promo_sales_filled_aggregate_transaction_to_sku`
  GROUP BY
    sku_root_id,
    store_id,
    description,
    area,
    section,
    category,
    subcategory,
    segment,
    brand_name,
    eroskibrand_flag,
    eroskibrand_label,
    date,
    promo_id,
    promo_year ),
  npr AS (
  SELECT
    sku_root_id,
    store_id,
    date,
    COUNT(*) AS n_promo
  FROM
    trans_to_sku
  WHERE
    promo_id IS NOT NULL
    AND promo_year IS NOT NULL
  GROUP BY
    sku_root_id,
    store_id,
    date ),
  npr_aggr_daily_transaction_to_sku AS (
  SELECT
    trans.*,
    IFNULL(npr.n_promo,
      0) AS n_promo
  FROM
    trans_to_sku trans
  LEFT JOIN
    npr
  ON
    npr.sku_root_id = trans.sku_root_id
    AND npr.store_id = trans.store_id
    AND npr.date = trans.date )
SELECT
  *,
  CASE
    WHEN n_promo = 0 THEN 0
  ELSE
  1
END
  AS promo_flag
FROM
  npr_aggr_daily_transaction_to_sku;
  
  # Create a daily aggregate transaction to sku
  CREATE OR REPLACE TABLE
  `gum-eroski-dev.ETL.aggregate_daily_transaction_to_sku`
PARTITION BY
  date
CLUSTER BY
  sku_root_id,
  store_id,
  category,
  promo_flag AS
  SELECT
  sku_root_id,
  store_id,
  description,
  area,
  section,
  category,
  subcategory,
  segment,
  brand_name,
  eroskibrand_flag,
  eroskibrand_label,
  date,
  ARRAY_AGG(distinct promo_id IGNORE NULLS) as promo_id_list,
  ARRAY_AGG(distinct promo_year IGNORE NULLS) as promo_year_list,
  SUM(total_price_if_sku_std_price) as total_price_if_sku_std_price,
  AVG(std_price_per_unit) as std_price_per_unit,
  SUM(total_sale_amt_std) as total_sale_amt_std,
  SUM(total_sale_amt_discount) as total_sale_amt_discount,
  SUM(total_sale_amt_markdown) as total_sale_amt_markdown,
  SUM(total_sale_amt_campaign) as total_sale_amt_campaign,
  SUM(total_sale_amt_other) as total_sale_amt_other,
  SUM(total_sale_amt) as total_sale_amt,
  SUM(total_sale_qty_std) as total_sale_qty_std,
  SUM(total_sale_qty_discount) as total_sale_qty_discount,
  SUM(total_sale_qty_markdown) as total_sale_qty_markdown,
  SUM(total_sale_qty_campaign) as total_sale_qty_campaign,
  SUM(total_sale_qty_other) as total_sale_qty_other,
  SUM(total_sale_qty) as total_sale_qty,
  SUM(total_margin_amt_std) as total_margin_amt_std,
  SUM(total_margin_amt_discount) as total_margin_amt_discount,
  SUM(total_margin_amt_markdown) as total_margin_amt_markdown,
  SUM(total_margin_amt_campaign) as total_margin_amt_campaign,
  SUM(total_margin_amt_other) as total_margin_amt_other,
  SUM(total_margin_amt) as total_margin_amt,
  SUM(total_non_oferta_discount_amt) as total_non_oferta_discount_amt,
  SUM(total_non_oferta_discount_amt_erk) as total_non_oferta_discount_amt_erk,
  SUM(oferta_promo_total_sale_amt) as oferta_promo_total_sale_amt,
  SUM(oferta_promo_total_sale_qty) as oferta_promo_total_sale_qty,
  SUM(oferta_promo_total_margin_amt) as oferta_promo_total_margin_amt,
  SUM(oferta_promo_total_discount_amt) as oferta_promo_total_discount_amt,
  MAX(n_promo) as n_promo,
  MAX(promo_flag) as promo_flag
  FROM
    `gum-eroski-dev.ETL.aggregate_daily_transaction_to_sku_promo` 
  GROUP BY 
  sku_root_id,
  store_id,
  description,
  area,
  section,
  category,
  subcategory,
  segment,
  brand_name,
  eroskibrand_flag,
  eroskibrand_label,
  date;
  
  # Create a weekly aggregate transaction to sku promo table
CREATE OR REPLACE TABLE
  `ETL.aggregate_weekly_transaction_to_sku_promo`
PARTITION BY
  date
CLUSTER BY
  sku_root_id,
  store_id,
  area,
  promo_flag AS
WITH weekly_aggr AS ( 
SELECT
  DATE_TRUNC(date, WEEK(MONDAY)) AS date,
  #get first Monday of week
  sku_root_id,
  store_id,
  description,
  area,
  section,
  category,
  subcategory,
  segment,
  brand_name,
  eroskibrand_flag,
  eroskibrand_label,
  promo_id,
  promo_year,
  SUM(total_sale_amt_std) AS total_sale_amt_std,
  SUM(total_sale_amt_discount) AS total_sale_amt_discount,
  SUM(total_sale_amt_markdown) AS total_sale_amt_markdown,
  SUM(total_sale_amt_campaign) AS total_sale_amt_campaign,
  SUM(total_sale_amt_other) AS total_sale_amt_other,
  SUM(total_sale_amt) AS total_sale_amt,
  SUM(total_sale_qty_std) AS total_sale_qty_std,
  SUM(total_sale_qty_discount) AS total_sale_qty_discount,
  SUM(total_sale_qty_markdown) AS total_sale_qty_markdown,
  SUM(total_sale_qty_campaign) AS total_sale_qty_campaign,
  SUM(total_sale_qty_other) AS total_sale_qty_other,
  SUM(total_sale_qty) AS total_sale_qty,
  SUM(total_margin_amt_std) AS total_margin_amt_std,
  SUM( total_margin_amt_discount) AS total_margin_amt_discount,
  SUM( total_margin_amt_markdown) AS total_margin_amt_markdown,
  SUM( total_margin_amt_campaign) AS total_margin_amt_campaign,
  SUM( total_margin_amt_other) AS total_margin_amt_other,
  SUM( total_margin_amt) AS total_margin_amt,
  SUM(total_price_if_sku_std_price) AS total_price_if_sku_std_price,
  AVG(std_price_per_unit) AS std_price_per_unit,
  SUM(total_non_oferta_discount_amt) AS total_non_oferta_discount_amt,
  SUM(total_non_oferta_discount_amt_erk) AS total_non_oferta_discount_amt_erk,
  SUM(oferta_promo_total_sale_amt) AS oferta_promo_total_sale_amt,
  SUM(oferta_promo_total_sale_qty) AS oferta_promo_total_sale_qty,
  SUM(oferta_promo_total_margin_amt) AS oferta_promo_total_margin_amt,
  SUM(oferta_promo_total_discount_amt) AS oferta_promo_total_discount_amt,
  SUM(CASE
      WHEN promo_id is not null THEN promo_flag
    ELSE
    0
  END
    ) AS no_days_with_promo
FROM
  `gum-eroski-dev.ETL.aggregate_daily_transaction_to_sku_promo`
GROUP BY
  DATE_TRUNC(date, WEEK(MONDAY)),
  sku_root_id,
  store_id,
  description,
  area,
  section,
  category,
  subcategory,
  segment,
  brand_name,
  eroskibrand_flag,
  eroskibrand_label,
  promo_id,
  promo_year
),   npr AS (
  SELECT
    sku_root_id,
    store_id,
    date,
    COUNT(*) AS n_promo
  FROM
    weekly_aggr
  WHERE
    promo_id IS NOT NULL
    AND promo_year IS NOT NULL
  GROUP BY
    sku_root_id,
    store_id,
    date ),
  npr_aggr_weekly_transaction_to_sku AS (
  SELECT
    trans.*,
    IFNULL(npr.n_promo,
      0) AS n_promo
  FROM
    weekly_aggr trans
  LEFT JOIN
    npr
  ON
    npr.sku_root_id = trans.sku_root_id
    AND npr.store_id = trans.store_id
    AND npr.date = trans.date )
SELECT
  *,
  CASE
    WHEN n_promo = 0 THEN 0
  ELSE
  1
END
  AS promo_flag
FROM
  npr_aggr_weekly_transaction_to_sku;
  
  ##create a aggregate weekly transaction table
CREATE OR REPLACE TABLE
  `ETL.aggregate_weekly_transaction_to_sku`
PARTITION BY
  date
CLUSTER BY
  sku_root_id,
  store_id,
  area,
  promo_flag AS
SELECT
  sku_root_id,
  store_id,
  description,
  area,
  section,
  category,
  subcategory,
  segment,
  brand_name,
  eroskibrand_flag,
  eroskibrand_label,
  date,
  ARRAY_AGG(distinct promo_id IGNORE NULLS) as promo_id_list,
  ARRAY_AGG(distinct promo_year IGNORE NULLS) as promo_year_list,
  SUM(total_price_if_sku_std_price) as total_price_if_sku_std_price,
  AVG(std_price_per_unit) as std_price_per_unit,
  SUM(total_sale_amt_std) as total_sale_amt_std,
  SUM(total_sale_amt_discount) as total_sale_amt_discount,
  SUM(total_sale_amt_markdown) as total_sale_amt_markdown,
  SUM(total_sale_amt_campaign) as total_sale_amt_campaign,
  SUM(total_sale_amt_other) as total_sale_amt_other,
  SUM(total_sale_amt) as total_sale_amt,
  SUM(total_sale_qty_std) as total_sale_qty_std,
  SUM(total_sale_qty_discount) as total_sale_qty_discount,
  SUM(total_sale_qty_markdown) as total_sale_qty_markdown,
  SUM(total_sale_qty_campaign) as total_sale_qty_campaign,
  SUM(total_sale_qty_other) as total_sale_qty_other,
  SUM(total_sale_qty) as total_sale_qty,
  SUM(total_margin_amt_std) as total_margin_amt_std,
  SUM(total_margin_amt_discount) as total_margin_amt_discount,
  SUM(total_margin_amt_markdown) as total_margin_amt_markdown,
  SUM(total_margin_amt_campaign) as total_margin_amt_campaign,
  SUM(total_margin_amt_other) as total_margin_amt_other,
  SUM(total_margin_amt) as total_margin_amt,
  SUM(total_non_oferta_discount_amt) as total_non_oferta_discount_amt,
  SUM(total_non_oferta_discount_amt_erk) as total_non_oferta_discount_amt_erk,
  SUM(oferta_promo_total_sale_amt) as oferta_promo_total_sale_amt,
  SUM(oferta_promo_total_sale_qty) as oferta_promo_total_sale_qty,
  SUM(oferta_promo_total_margin_amt) as oferta_promo_total_margin_amt,
  SUM(oferta_promo_total_discount_amt) as oferta_promo_total_discount_amt,
  MAX(n_promo) as n_promo,
  MAX(promo_flag) as promo_flag,
  MAX(no_days_with_promo) as no_days_with_promo
  FROM
    `gum-eroski-dev.ETL.aggregate_weekly_transaction_to_sku_promo` 
  GROUP BY 
  sku_root_id,
  store_id,
  description,
  area,
  section,
  category,
  subcategory,
  segment,
  brand_name,
  eroskibrand_flag,
  eroskibrand_label,
  date;
  
 ##Create aggregate daily transaction to sku summary
CREATE OR REPLACE TABLE
  `ETL.aggregate_daily_transaction_summary`
PARTITION BY
  date
CLUSTER BY
  sku_root_id,
  area,
  section,
  promo_flag AS
  SELECT
    date,
    area,
    section,
    category,
    subcategory,
    segment,
    brand_name,
    eroskibrand_flag,
    eroskibrand_label,
    sku_root_id,
    description,
    MAX(promo_flag) AS promo_flag,
    SUM(total_sale_amt) AS total_sale_amt,
    SUM(total_sale_qty) AS total_sale_qty,
    SUM(total_margin_amt) AS total_margin_amt,
    SUM(total_price_if_sku_std_price) AS total_price_if_sku_std_price,
    AVG(std_price_per_unit) AS std_price_per_unit,
    SUM(total_non_oferta_discount_amt) AS total_non_oferta_discount_amt,
    SUM(total_non_oferta_discount_amt_erk) AS total_non_oferta_discount_amt_erk,
    SUM(oferta_promo_total_sale_amt) AS oferta_promo_total_sale_amt,
    SUM(oferta_promo_total_sale_qty) AS oferta_promo_total_sale_qty,
    SUM(oferta_promo_total_margin_amt) AS oferta_promo_total_margin_amt,
    SUM(oferta_promo_total_discount_amt) AS oferta_promo_total_discount_amt
  FROM
    `gum-eroski-dev.ETL.aggregate_daily_transaction_to_sku`
  GROUP BY
    date,
    area,
    section,
    category,
    subcategory,
    segment,
    brand_name,
    eroskibrand_flag,
    eroskibrand_label,
    sku_root_id,
    description;
 ##Create aggregate weekly transaction to sku summary
CREATE OR REPLACE TABLE
`ETL.aggregate_weekly_transaction_summary`
PARTITION BY
  date
CLUSTER BY
  sku_root_id,
  area,
  section,
  promo_flag AS
  SELECT
    date,
    area,
    section,
    category,
    subcategory,
    segment,
    brand_name,
    eroskibrand_flag,
    eroskibrand_label,
    sku_root_id,
    description,
    MAX(promo_flag) AS promo_flag,
    SUM(total_sale_amt) AS total_sale_amt,
    SUM(total_sale_qty) AS total_sale_qty,
    SUM(total_margin_amt) AS total_margin_amt,
    SUM(total_price_if_sku_std_price) AS total_price_if_sku_std_price,
    AVG(std_price_per_unit) AS std_price_per_unit,
    SUM(total_non_oferta_discount_amt) AS total_non_oferta_discount_amt,
    SUM(total_non_oferta_discount_amt_erk) AS total_non_oferta_discount_amt_erk,
    SUM(oferta_promo_total_sale_amt) AS oferta_promo_total_sale_amt,
    SUM(oferta_promo_total_sale_qty) AS oferta_promo_total_sale_qty,
    SUM(oferta_promo_total_margin_amt) AS oferta_promo_total_margin_amt,
    SUM(oferta_promo_total_discount_amt) AS oferta_promo_total_discount_amt
  FROM
    `gum-eroski-dev.ETL.aggregate_weekly_transaction_to_sku`
  GROUP BY
    date,
    area,
    section,
    category,
    subcategory,
    segment,
    brand_name,
    eroskibrand_flag,
    eroskibrand_label,
    sku_root_id,
    description;
  # Create a agg std price margin per sku table
CREATE OR REPLACE TABLE
  ETL.aggregate_std_price_margin AS
WITH
  tran_level AS(
  SELECT
    sku_root_id,
    AVG(safe_divide(total_margin_amt,
        total_sale_qty)) AS margin_per_unit,
    AVG(std_price_per_unit) AS std_price_per_unit,
    AVG(safe_divide(total_sale_amt,
        total_sale_qty)) AS std_price_per_unit_2
  FROM
    `ETL.transaction_to_sku`
  INNER JOIN (
    SELECT
      DISTINCT sku_root_id
    FROM
      ETL.root_sku)
  USING
    (sku_root_id)
  WHERE
    promo_id IS NULL
  GROUP BY
    sku_root_id ),
  aggdaily_level AS(
  SELECT
    sku_root_id,
    AVG(safe_divide(total_margin_amt,
        total_sale_qty)) AS margin_per_unit,
    AVG(std_price_per_unit) AS std_price_per_unit,
    AVG(safe_divide(total_sale_amt,
        total_sale_qty)) AS std_price_per_unit_2
  FROM
    `ETL.aggregate_daily_transaction_to_sku`
  INNER JOIN (
    SELECT
      DISTINCT sku_root_id
    FROM
      ETL.root_sku)
  USING
    (sku_root_id)
  WHERE
    promo_flag = 0
  GROUP BY
    sku_root_id ),
  t1 AS (
  SELECT
    sku_root_id,
    margin_per_unit,
    CASE
      WHEN std_price_per_unit =0 OR std_price_per_unit IS NULL THEN std_price_per_unit_2
    ELSE
    std_price_per_unit
  END
    AS std_price_per_unit
  FROM
    aggdaily_level ),
  t2 AS (
  SELECT
    sku_root_id,
    margin_per_unit,
    CASE
      WHEN std_price_per_unit =0 OR std_price_per_unit IS NULL THEN std_price_per_unit_2
    ELSE
    std_price_per_unit
  END
    AS std_price_per_unit
  FROM
    tran_level),
  table AS(
  SELECT
    t1.sku_root_id,
    t1.margin_per_unit,
    t1.std_price_per_unit,
    t2.margin_per_unit AS margin,
    t2.std_price_per_unit AS price1
  FROM
    t1
  LEFT JOIN
    t2
  USING
    (sku_root_id)),
  b_final AS (
  SELECT
    sku_root_id,
    CASE
      WHEN margin_per_unit = 0 OR margin_per_unit IS NULL THEN margin
    ELSE
    margin_per_unit
  END
    AS margin_per_unit,
    CASE
      WHEN std_price_per_unit =0 OR std_price_per_unit IS NULL THEN price1
    ELSE
    std_price_per_unit
  END
    AS std_price_per_unit
  FROM
    table)
SELECT
  *,
  (std_price_per_unit-margin_per_unit) AS cost_per_unit
FROM
  b_final ;
##Create a temp total aggregate promo to sku table
CREATE OR REPLACE TABLE
  `ETL.temp_daily_total_aggregate_promo_to_sku`
PARTITION BY
  date
CLUSTER BY
  promo_id,
  store_id,
  sku_root_id,
  promo_mechanic OPTIONS( expiration_timestamp=TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY) ) AS
WITH
  promo_campaign_cal AS (
  SELECT
    promo_id,
    promo_year,
    name,
    type,
    start_date,
    end_date,
    class,
    customer_profile_type,
    marketing_type,
    DATE_DIFF(end_date, start_date, DAY)+1 AS duration,
    cal.date,
    cal.weekend,
    cal_start_date.day,
    cal_start_date.month,
    cal_start_date.quarter,
    cal_start_date.week
  FROM
    `ETL.promotional_campaign`
  INNER JOIN
    `ETL.calendar` cal
  ON
    cal.date BETWEEN start_date
    AND end_date
  INNER JOIN
    `ETL.calendar` cal_start_date
  ON
    cal_start_date.date = start_date ),
  p_promo_to_sku AS (
    SELECT
    promo.sku_root_id,
    promo.store_id,
    promo.promo_mechanic,
    promo.promo_mechanic_description,
    MAX(promo.std_price) as std_price,
    MIN(promo.discounted_price_promo) as discounted_price_promo,
	  promo.discount_depth_rank,
    promo.discount_depth,
    MAX(promo.leaflet_cover) AS leaflet_cover,
    MAX(promo.leaflet_priv_space) AS leaflet_priv_space,
    MAX(promo.in_leaflet_flag) AS in_leaflet_flag,
    MAX(promo.in_gondola_flag) AS in_gondola_flag,
    MAX(promo.in_both_leaflet_gondola_flag) AS in_both_leaflet_gondola_flag,
    promo.promo_id,
    promo.promo_year
  FROM
    `ETL.promotional_to_sku` promo
  
  WHERE ((promo_mechanic in ('10','20','30','40','60') AND (discount_depth is not NULL OR discounted_price_promo is not NULL)) 
  OR (promo_mechanic not in ('10','20','30','40','60')))
  GROUP BY 
    promo.sku_root_id,
    promo.store_id,
    promo.promo_mechanic,
    promo.promo_mechanic_description,
    promo.discount_depth_rank,
    promo.discount_depth,
    promo.promo_id,
    promo.promo_year
  ), 
  promo_to_sku AS (
  SELECT
    promo.sku_root_id,
    root_sku_t.description,
    root_sku_t.area,
    root_sku_t.section,
    root_sku_t.category,
    root_sku_t.subcategory,
    root_sku_t.segment,
    root_sku_t.brand_name,
    root_sku_t.eroskibrand_flag,
    root_sku_t.eroskibrand_label,
    root_sku_t.wealthy_range_flag,
    root_sku_t.flag_healthy,
    root_sku_t.innovation_flag,
    root_sku_t.tourism_flag,
    root_sku_t.local_flag,
    root_sku_t.regional_flag,
    root_sku_t.wow_flag,
    promo.store_id,
    store.store_geography_region,
    store.store_type1 AS store_format,
    store.size_m2 AS store_size,
    promo.promo_mechanic,
    promo.promo_mechanic_description,
    AVG(promo.std_price) as std_price,
    promo.discounted_price_promo,
	promo.discount_depth_rank,
    promo.discount_depth,
    MAX(promo.leaflet_cover) AS leaflet_cover,
    MAX(promo.leaflet_priv_space) AS leaflet_priv_space,
    MAX(promo.in_leaflet_flag) AS in_leaflet_flag,
    MAX(promo.in_gondola_flag) AS in_gondola_flag,
    MAX(promo.in_both_leaflet_gondola_flag) AS in_both_leaflet_gondola_flag,
    promo.promo_id,
    promo.promo_year
  FROM
   p_promo_to_sku promo
  INNER JOIN
    `ETL.store` store
  ON
    store.store_id = promo.store_id
  INNER JOIN
    `ETL.root_sku` root_sku_t
  ON
    root_sku_t.sku_root_id = promo.sku_root_id
  GROUP BY
    promo.sku_root_id,
    promo.store_id,
    promo.promo_mechanic,
    promo.promo_mechanic_description,
    promo.discounted_price_promo,
	promo.discount_depth_rank,
    promo.discount_depth,
    promo.promo_id,
    promo.promo_year,
    store.store_geography_region,
    store.store_type1,
    store.size_m2,
    root_sku_t.description,
    root_sku_t.area,
    root_sku_t.section,
    root_sku_t.category,
    root_sku_t.subcategory,
    root_sku_t.segment,
    root_sku_t.brand_name,
    root_sku_t.eroskibrand_flag,
    root_sku_t.eroskibrand_label,
    root_sku_t.wealthy_range_flag,
    root_sku_t.flag_healthy,
    root_sku_t.innovation_flag,
    root_sku_t.tourism_flag,
    root_sku_t.local_flag,
    root_sku_t.regional_flag,
    root_sku_t.wow_flag )
  SELECT
    promo.*,
    promo_to_sku.* EXCEPT (promo_id,
      promo_year,
      std_price),
    CASE
      WHEN promo_to_sku.std_price IS NULL THEN trans.std_price_per_unit
    ELSE
    promo_to_sku.std_price
  END
    AS std_price,
  FROM
    promo_to_sku promo_to_sku
  INNER JOIN
    promo_campaign_cal promo
  ON
    promo_to_sku.promo_id = promo.promo_id
    AND promo_to_sku.promo_year = promo.promo_year
  LEFT JOIN
    `ETL.aggregate_daily_transaction_to_sku` trans
  ON
    trans.date = promo.date
    AND trans.sku_root_id = promo_to_sku.sku_root_id
    AND trans.store_id = promo_to_sku.store_id;
	
	
   ##Create a total aggregate promo to sku table
CREATE OR REPLACE TABLE
  `ETL.temp_trans_daily_total_aggregate_promo_to_sku`
PARTITION BY
  date
CLUSTER BY
  promo_id,
  store_id,
  sku_root_id,
  promo_mechanic OPTIONS( expiration_timestamp=TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY) ) AS
WITH
  npr_mech_dd AS (
SELECT
    promo_id, 
	promo_year,
	sku_root_id,
    store_id,
    date,
    COUNT(*) AS n_promo_mech
  FROM
    `ETL.temp_daily_total_aggregate_promo_to_sku`
    GROUP BY 
    promo_id, 
	promo_year,
	sku_root_id,
    store_id,
    date), 
    promo_aggr AS (
    SELECT promo.*,
    npr_mech_dd.n_promo_mech
    FROM
    `ETL.temp_daily_total_aggregate_promo_to_sku` promo
    LEFT JOIN  npr_mech_dd
    ON npr_mech_dd.promo_id = promo.promo_id
	  AND npr_mech_dd.promo_year = promo.promo_year
	  AND npr_mech_dd.sku_root_id = promo.sku_root_id
    AND npr_mech_dd.store_id = promo.store_id
    AND npr_mech_dd.date = promo.date
    )
  SELECT
    promo.*,
    CASE
      WHEN promo.promo_mechanic = '10' AND 
	  promo.discounted_price_promo > 0 AND 
	  promo.std_price > 0 AND 
	  promo.std_price >= promo.discounted_price_promo 
      THEN 100*(1-SAFE_DIVIDE(promo.discounted_price_promo, promo.std_price))
    ELSE
    NULL
  END
    AS discount_depth_10,
	
  trans_promo.total_sale_amt as pr_total_sale_amt,
  trans_promo.total_sale_qty as pr_total_sale_qty,
  trans_promo.total_margin_amt as pr_total_margin_amt,
  
  trans_non_promo.total_sale_amt as npr_total_sale_amt,
  trans_non_promo.total_sale_qty as npr_total_sale_qty,
  trans_non_promo.total_margin_amt as npr_total_margin_amt,
  
  trans_daily.n_promo as n_promo,
  
  IFNULL(trans_promo.total_sale_amt*(SAFE_DIVIDE(1,n_promo_mech)),0) + IFNULL(trans_non_promo.total_sale_amt*(SAFE_DIVIDE(1,(n_promo_mech*trans_daily.n_promo))),0) as total_sale_amt,
  IFNULL(trans_promo.total_sale_qty*(SAFE_DIVIDE(1,n_promo_mech)),0) + IFNULL(trans_non_promo.total_sale_qty*(SAFE_DIVIDE(1,(n_promo_mech*trans_daily.n_promo))),0) as total_sale_qty,
  IFNULL(trans_promo.total_margin_amt*(SAFE_DIVIDE(1,n_promo_mech)),0) + IFNULL(trans_non_promo.total_margin_amt*(SAFE_DIVIDE(1,(n_promo_mech*trans_daily.n_promo))),0) as total_margin_amt,
 
  (IFNULL(trans_promo.total_sale_amt*(SAFE_DIVIDE(1,n_promo_mech)),0) + IFNULL(trans_non_promo.total_sale_amt*(SAFE_DIVIDE(1,(n_promo_mech*trans_daily.n_promo))),0)) - 
  (IFNULL(trans_promo.total_price_if_sku_std_price*(SAFE_DIVIDE(1,n_promo_mech)),0) + IFNULL(trans_non_promo.total_price_if_sku_std_price*(SAFE_DIVIDE(1,(n_promo_mech*trans_daily.n_promo))),0)) AS total_discount_amt,

  trans_promo.oferta_promo_total_sale_amt,
  trans_promo.oferta_promo_total_sale_qty, 
  trans_promo.oferta_promo_total_margin_amt,
  trans_promo.oferta_promo_total_discount_amt	
  FROM
    promo_aggr promo
	LEFT JOIN 
	 `ETL.aggregate_daily_transaction_to_sku_promo` trans_promo
	ON
    trans_promo.date = promo.date
    AND trans_promo.sku_root_id = promo.sku_root_id
    AND trans_promo.store_id = promo.store_id
	AND trans_promo.promo_id = promo.promo_id
	AND trans_promo.promo_year = promo.promo_year
	
	LEFT JOIN 
	 `ETL.aggregate_daily_transaction_to_sku_promo` trans_non_promo
	ON
    trans_non_promo.date = promo.date
    AND trans_non_promo.sku_root_id = promo.sku_root_id
    AND trans_non_promo.store_id = promo.store_id
	AND trans_non_promo.promo_id is NULL
	AND trans_non_promo.promo_year is NULL
  
  LEFT JOIN 
	 `ETL.aggregate_daily_transaction_to_sku` trans_daily
	ON
    trans_daily.date = promo.date
    AND trans_daily.sku_root_id = promo.sku_root_id
    AND trans_daily.store_id = promo.store_id;
	
	
	
	# Remove all sku_root_ids and store_ids that are not in the transaction table
CREATE OR REPLACE TABLE
  `ETL.temp_trans_daily_total_aggregate_promo_to_sku_filtered`
PARTITION BY
  date
CLUSTER BY
  promo_id,
  store_id,
  sku_root_id,
  promo_mechanic OPTIONS( expiration_timestamp=TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY) ) AS
	SELECT *
  FROM
    `ETL.temp_trans_daily_total_aggregate_promo_to_sku`
  where n_promo is not NULL;

# Get the total discount depth per sku and store and promo
CREATE OR REPLACE TABLE
  `ETL.temp_trans_store_aggregate_promo_to_sku_summary`
 AS
 SELECT 
 promo_id, 
 promo_year, 
 name, 
 sku_root_id, 
 store_id,
 promo_mechanic,
 discount_depth_rank,
 discount_depth,
 AVG(discounted_price_promo) as discounted_price_promo,
 AVG(std_price) as std_price,
 AVG(discount_depth_10) as discount_depth_10
 
 FROM `ETL.temp_trans_daily_total_aggregate_promo_to_sku_filtered`
 
 WHERE promo_mechanic = '10'
 
 group by 
 promo_id, 
 promo_year, 
 name, 
 sku_root_id, 
 store_id,
 promo_mechanic,
 discount_depth_rank,
 discount_depth; 
 
 
 # Get the total discount depth per sku and promo
CREATE OR REPLACE TABLE
  `ETL.temp_trans_total_aggregate_promo_to_sku_summary`
  OPTIONS( expiration_timestamp=TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY) )
AS 
WITH avg_dd AS (
SELECT 
 promo_id, 
 promo_year, 
 name, 
 sku_root_id, 
 promo_mechanic,
 discount_depth_rank,
 discount_depth,
 AVG(discounted_price_promo) as discounted_price_promo,
 AVG(std_price) as std_price,
 AVG(discount_depth_10) as discount_depth_10
 
 FROM `ETL.temp_trans_store_aggregate_promo_to_sku_summary`
 
 group by 
 promo_id, 
 promo_year, 
 name, 
 sku_root_id, 
 promo_mechanic,
 discount_depth_rank,
 discount_depth

)
 SELECT 
 promo_id, 
 promo_year, 
 name, 
 sku_root_id, 
 promo_mechanic,
 CASE
    WHEN discount_depth_10 IS NOT NULL AND discount_depth_10>5 THEN CONCAT(CAST(ROUND(discount_depth_10/5,0)*5 AS STRING),"% off")
    WHEN discount_depth_10 IS NOT NULL
  AND discount_depth_10>=0
  AND discount_depth_10<=5 THEN CONCAT(CAST(ROUND(discount_depth_10/2.5,0)*2.5 AS STRING),"% off")
  ELSE
  discount_depth
END
  AS discount_depth,
  CASE
    WHEN discount_depth_10 IS NOT NULL AND discount_depth_10>5 THEN ROUND(discount_depth_10/5,0)*5
    WHEN discount_depth_10 IS NOT NULL
  AND discount_depth_10>=0
  AND discount_depth_10<=5 THEN ROUND(discount_depth_10/2.5,0)*2.5
  ELSE discount_depth_rank
END
  AS discount_depth_rank,
 discounted_price_promo,
 std_price,
 discount_depth_10
 
 FROM avg_dd; 
 
 CREATE OR REPLACE TABLE
  `ETL.daily_total_aggregate_promo_to_sku`
PARTITION BY
  date
CLUSTER BY
  promo_id,
  store_id,
  sku_root_id,
  promo_mechanic AS
SELECT 
daily_promo.promo_id,
daily_promo.promo_year,
daily_promo.name,
daily_promo.type,
daily_promo.start_date,
daily_promo.end_date,
daily_promo.class,
daily_promo.customer_profile_type,
daily_promo.marketing_type,
daily_promo.duration,
daily_promo.date,
daily_promo.weekend,
daily_promo.day AS campaign_start_day,
daily_promo.month AS campaign_start_month,
daily_promo.quarter AS campaign_start_quarter,
daily_promo.week AS campaign_start_week,
daily_promo.sku_root_id,
daily_promo.description,
daily_promo.area,
daily_promo.section,
daily_promo.category,
daily_promo.subcategory,
daily_promo.segment,
daily_promo.brand_name,
daily_promo.eroskibrand_flag,
daily_promo.eroskibrand_label,
daily_promo.wealthy_range_flag,
daily_promo.flag_healthy,
daily_promo.innovation_flag,
daily_promo.tourism_flag,
daily_promo.local_flag,
daily_promo.regional_flag,
daily_promo.wow_flag,
daily_promo.store_id,
daily_promo.store_geography_region,
daily_promo.store_format,
daily_promo.store_size,
daily_promo.promo_mechanic,
daily_promo.promo_mechanic_description,
CASE WHEN daily_promo.promo_mechanic='10' THEN aggr_summary.discount_depth
ELSE daily_promo.discount_depth
END AS discount_depth,
CASE WHEN daily_promo.promo_mechanic='10' THEN aggr_summary.discount_depth_rank
ELSE daily_promo.discount_depth_rank
END AS discount_depth_rank,
daily_promo.leaflet_cover,
daily_promo.leaflet_priv_space,
daily_promo.in_leaflet_flag,
daily_promo.in_gondola_flag,
daily_promo.in_both_leaflet_gondola_flag,
daily_promo.n_promo_mech,
daily_promo.n_promo,
weekly_trans_prev.promo_flag as promo_prev_week,
weekly_trans_pf.promo_flag as promo_pf_week,
daily_promo.total_sale_amt,
daily_promo.total_sale_qty,
daily_promo.total_margin_amt,
daily_promo.total_discount_amt,
daily_promo.oferta_promo_total_sale_amt,
daily_promo.oferta_promo_total_sale_qty,
daily_promo.oferta_promo_total_margin_amt,
daily_promo.oferta_promo_total_discount_amt

FROM `ETL.temp_trans_daily_total_aggregate_promo_to_sku_filtered` daily_promo

LEFT JOIN `ETL.temp_trans_total_aggregate_promo_to_sku_summary` aggr_summary
ON aggr_summary.promo_id = daily_promo.promo_id
AND aggr_summary.promo_year = daily_promo.promo_year
AND aggr_summary.name = daily_promo.name
AND aggr_summary.sku_root_id = daily_promo.sku_root_id
AND aggr_summary.promo_mechanic = daily_promo.promo_mechanic

LEFT JOIN `ETL.aggregate_weekly_transaction_to_sku` weekly_trans_prev
ON weekly_trans_prev.date = DATE_SUB(DATE_TRUNC(daily_promo.start_date, WEEK(MONDAY)), INTERVAL 7 DAY)
AND weekly_trans_prev.sku_root_id = daily_promo.sku_root_id
AND weekly_trans_prev.store_id = daily_promo.store_id

LEFT JOIN `ETL.aggregate_weekly_transaction_to_sku` weekly_trans_pf
ON weekly_trans_pf.date = DATE_ADD(DATE_TRUNC(daily_promo.end_date, WEEK(MONDAY)), INTERVAL 7 DAY)
AND weekly_trans_pf.sku_root_id = daily_promo.sku_root_id
AND weekly_trans_pf.store_id = daily_promo.store_id;

CREATE OR REPLACE TABLE
  `ETL.weekly_total_aggregate_promo_to_sku`
PARTITION BY
  date
CLUSTER BY
  promo_id,
  store_id,
  sku_root_id,
  promo_mechanic AS
WITH daily_promo AS (
	SELECT *, 1 as promo_flag
	FROM
    `ETL.daily_total_aggregate_promo_to_sku`
)
SELECT
    DATE_TRUNC(promo.date, WEEK(MONDAY)) AS date,
    promo.promo_id,
    promo.promo_year,
    promo.name,
    promo.type,
    promo.start_date,
    promo.end_date,
    promo.class,
    promo.customer_profile_type,
    promo.marketing_type,
    promo.duration,
    MAX(promo.weekend) AS includes_weekend,
    promo.campaign_start_day,
    promo.campaign_start_month,
    promo.campaign_start_quarter,
    promo.campaign_start_week,
    promo.sku_root_id,
    promo.description,
    promo.area,
    promo.section,
    promo.category,
    promo.subcategory,
    promo.segment,
    promo.brand_name,
    promo.eroskibrand_flag,
    promo.eroskibrand_label,
    promo.wealthy_range_flag,
    promo.flag_healthy,
    promo.innovation_flag,
    promo.tourism_flag,
    promo.local_flag,
    promo.regional_flag,
    promo.wow_flag,
    promo.store_id,
    promo.store_geography_region,
    promo.store_format,
    promo.store_size,
    promo.promo_mechanic,
    promo.promo_mechanic_description,
    MAX(promo.leaflet_cover) AS leaflet_cover,
    MAX(promo.leaflet_priv_space) AS leaflet_priv_space,
    MAX(promo.in_leaflet_flag) AS in_leaflet_flag,
    MAX(promo.in_gondola_flag) AS in_gondola_flag,
    MAX(promo.in_both_leaflet_gondola_flag) AS in_both_leaflet_gondola_flag,
	promo.discount_depth_rank,
    promo.discount_depth,
	MAX(promo.n_promo_mech) as n_promo_mech,
	MAX(promo.n_promo) as n_promo,
	MAX(promo.promo_prev_week) as promo_prev_week,
	MAX(promo.promo_pf_week) as promo_pf_week,
	SUM(promo.promo_flag) as n_days_promo_week,
	SUM(promo.total_sale_amt) as total_sale_amt,
	SUM(promo.total_sale_qty) as total_sale_qty,
	SUM(promo.total_margin_amt) as total_margin_amt,
	SUM(promo.total_discount_amt) as total_discount_amt,
	SUM(promo.oferta_promo_total_sale_amt) as oferta_promo_total_sale_amt,
	SUM(promo.oferta_promo_total_sale_qty) as oferta_promo_total_sale_qty,
	SUM(promo.oferta_promo_total_margin_amt) as oferta_promo_total_margin_amt,
	SUM(promo.oferta_promo_total_discount_amt) as oferta_promo_total_discount_amt
	
  FROM
    daily_promo promo
  GROUP BY
    DATE_TRUNC(date, WEEK(MONDAY)),
    promo_id,
    promo_year,
    name,
    type,
    start_date,
    end_date,
    class,
    customer_profile_type,
    marketing_type,
    duration,
    campaign_start_day,
    campaign_start_month,
    campaign_start_quarter,
    campaign_start_week,
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
	discount_depth_rank,
    discount_depth,
    store_id,
    store_geography_region,
    store_format,
    store_size;
	
	
## create the promo_to_sku agg summary table
CREATE OR REPLACE TABLE
  `ETL.temp_aggregate_promo_to_sku_summary` 
PARTITION BY
  date
CLUSTER BY
  promo_id,
  promo_year,
  sku_root_id,
  promo_mechanic OPTIONS( expiration_timestamp=TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY) ) AS
WITH
avg_bline AS (
SELECT sku_root_id, store_id, avg(total_sale_qty) as avg_bline_qty FROM 
    
    `ETL.aggregate_weekly_transaction_to_sku` 
    WHERE promo_flag = 0
    GROUP BY sku_root_id, store_id

), avg_bline_all AS (

SELECT sku_root_id, store_id, avg(total_sale_qty) as avg_bline_qty_all FROM 
    
    `ETL.aggregate_weekly_transaction_to_sku` 
    GROUP BY sku_root_id, store_id


),  bline_start AS (
  SELECT
    *,
    DATE_SUB(DATE_TRUNC(start_date, WEEK(MONDAY)), INTERVAL 7 DAY) AS promo_bline_start_date,
    DATE_ADD(DATE_TRUNC(end_date, WEEK(MONDAY)), INTERVAL 7 DAY) AS promo_pf_start_date
  FROM
    `ETL.weekly_total_aggregate_promo_to_sku`
  ),  bline_qty AS (
  SELECT
    bline_start.*,
    CASE
      WHEN store_format = "HIPERMERCADOS" THEN 1
    ELSE
    0
  END
    AS hipermercados_flag,
    CASE
      WHEN store_format = "SUPERMERCADOS" THEN 1
    ELSE
    0
  END
    AS supermercados_flag,
    CASE
      WHEN store_format = "GASOLINERAS" THEN 1
    ELSE
    0
  END
    AS gasolineras_flag,
    CASE
      WHEN store_format = "COMERCIO ELECTRONICO" THEN 1
    ELSE
    0
  END
    AS comercio_electronico_flag,
    CASE
      WHEN store_format = "OTROS NEGOCIOS" THEN 1
    ELSE
    0
  END
    AS otros_negocios_flag,
    CASE
      WHEN store_format = "PLATAFORMAS" THEN 1
    ELSE
    0
  END
    AS plataformas_flag,
    CASE
      WHEN store_format = "69" THEN 1
    ELSE
    0
  END
    AS other_flag,
    trans.total_sale_qty AS s_prev_bl_qty,
	trans.total_sale_qty*SAFE_DIVIDE(bline_start.n_days_promo_week,7) AS s_prev_bl_qty_norm,
    trans_after.total_sale_qty AS pf_after_bl_qty,	
    avg_bline.avg_bline_qty*SAFE_DIVIDE(bline_start.n_days_promo_week,7) AS avg_bline_qty,
	  avg_bline_all.avg_bline_qty_all*SAFE_DIVIDE(bline_start.n_days_promo_week,7) AS avg_bline_qty_all
  FROM
    bline_start
  LEFT JOIN `ETL.aggregate_weekly_transaction_to_sku` trans
  ON
    trans.date = promo_bline_start_date
    AND trans.sku_root_id = bline_start.sku_root_id
    AND trans.store_id = bline_start.store_id 
    
    LEFT JOIN `ETL.aggregate_weekly_transaction_to_sku` trans_after
  ON
    trans_after.date = promo_pf_start_date
    AND trans_after.sku_root_id = bline_start.sku_root_id
    AND trans_after.store_id = bline_start.store_id 
	
    LEFT JOIN avg_bline
  ON   avg_bline.sku_root_id = bline_start.sku_root_id
    AND avg_bline.store_id = bline_start.store_id 
	
	LEFT JOIN avg_bline_all
  ON   avg_bline_all.sku_root_id = bline_start.sku_root_id
    AND avg_bline_all.store_id = bline_start.store_id 
      
    )
  SELECT
    date,
    promo_id,
    promo_year,
    name,
    type,
    start_date,
    end_date,
    class,
    customer_profile_type,
    marketing_type,
    duration,
    MAX(includes_weekend) AS includes_weekend,
    campaign_start_day,
    campaign_start_month,
    campaign_start_quarter,
    campaign_start_week,
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
    SUM(hipermercados_flag) AS no_hipermercados_stores,
    SUM(supermercados_flag) AS no_supermercados_stores,
    SUM(gasolineras_flag) AS no_gasolineras_stores,
    SUM(comercio_electronico_flag) AS no_comercio_electronico_stores,
    SUM(otros_negocios_flag) AS no_otros_negocio_stores,
    SUM(plataformas_flag) AS no_plataformas_stores,
    SUM(other_flag) AS no_other_stores,
    ARRAY_AGG(DISTINCT store_id) AS store_ids,
    COUNT(DISTINCT store_id) AS no_impacted_stores,
    COUNT(DISTINCT store_geography_region) AS no_impacted_regions,
    AVG(store_size) AS avg_store_size,
    promo_mechanic,
    promo_mechanic_description,
    MAX(leaflet_cover) AS leaflet_cover,
    MAX(leaflet_priv_space) AS leaflet_priv_space,
    MAX(in_leaflet_flag) AS in_leaflet_flag,
    MAX(in_gondola_flag) AS in_gondola_flag,
    MAX(in_both_leaflet_gondola_flag) AS in_both_leaflet_gondola_flag,
	  discount_depth_rank,
    discount_depth,
    MAX(n_promo_mech) AS n_promo_mech,
    MAX(n_promo) AS n_promo,
    MAX(promo_prev_week) AS promo_prev_week,
    MAX(promo_pf_week) AS promo_pf_week,
    AVG(n_days_promo_week) as n_days_promo_week,
    SUM(total_sale_amt) AS total_sale_amt,
    SUM(total_sale_qty) AS total_sale_qty,
    SUM(total_margin_amt) AS total_margin_amt,
    SUM(total_discount_amt) AS total_discount_amt,	
	  SUM(oferta_promo_total_sale_amt) AS oferta_promo_total_sale_amt,
    SUM(oferta_promo_total_sale_qty) AS oferta_promo_total_sale_qty,
    SUM(oferta_promo_total_margin_amt) AS oferta_promo_total_margin_amt,
    SUM(oferta_promo_total_discount_amt) AS oferta_promo_total_discount_amt,
    SUM(s_prev_bl_qty) AS s_prev_bl_qty,
	SUM(s_prev_bl_qty_norm) AS s_prev_bl_qty_norm,
    SUM(pf_after_bl_qty) AS pf_after_bl_qty,
    SUM(avg_bline_qty) AS avg_bline_qty,
	  SUM(avg_bline_qty_all) AS avg_bline_qty_all
  FROM
    bline_qty
  GROUP BY
    date,
    promo_id,
    promo_year,
    name,
    type,
    start_date,
    end_date,
    class,
    customer_profile_type,
    marketing_type,
    duration,
    campaign_start_day,
    campaign_start_month,
    campaign_start_quarter,
    campaign_start_week,
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
	discount_depth_rank;	
  
  ## Create the promo_to_sku agg summary table
CREATE OR REPLACE TABLE
  `ETL.aggregate_promo_to_sku_summary`
PARTITION BY
  date
CLUSTER BY
  promo_id,
  type,
  sku_root_id,
  promo_mechanic AS

SELECT * FROM `gum-eroski-dev.ETL.temp_aggregate_promo_to_sku_summary`

where discount_depth is not null;

CREATE OR REPLACE TABLE
  `ETL.daily_total_aggregate_promo_to_sku_pf`
PARTITION BY
  date
CLUSTER BY
  promo_id,
  store_id,
  sku_root_id,
  promo_mechanic AS 
WITH d_promo AS (
SELECT distinct 
promo_id, 
promo_year, 
name, 
type, 
start_date, 
end_date, 
class,
customer_profile_type,
marketing_type,
duration,
campaign_start_day,
campaign_start_month,
campaign_start_quarter,
campaign_start_week,
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
store_id,
store_geography_region,
store_format,
store_size,
promo_mechanic,
promo_mechanic_description,
discount_depth,
discount_depth_rank,
leaflet_cover,
leaflet_priv_space,
in_leaflet_flag,
in_gondola_flag,
in_both_leaflet_gondola_flag,
n_promo_mech,
promo_prev_week

FROM `gum-eroski-dev.ETL.daily_total_aggregate_promo_to_sku`

)
SELECT trans.date,
promo.*, 
trans.total_sale_amt as pf_sale_amt, 
trans.total_sale_qty as pf_sale_qty,
trans.total_margin_amt as pf_margin_amt,
trans.n_promo as pf_n_promo,
trans.promo_flag as pf_promo_flag

FROM d_promo promo 
LEFT JOIN `ETL.aggregate_daily_transaction_to_sku` trans
ON trans.date > promo.end_date
AND trans.date <= DATE_ADD(promo.end_date, INTERVAL 28 DAY) 
AND trans.sku_root_id = promo.sku_root_id
AND trans.store_id = promo.store_id
WHERE trans.date is not null;


  ##Create SKU campaign dates table
CREATE OR REPLACE TABLE
  `ETL.sku_campaign_dates` AS
SELECT
  DISTINCT pr_campaign.* EXCEPT (type,
    class,
    customer_profile_type,
    marketing_type),
  CASE EXTRACT(DAYOFWEEK
  FROM
    pr_campaign.start_date)
    WHEN 1 THEN 'Sunday'
    WHEN 2 THEN 'Monday'
    WHEN 3 THEN 'Tuesday'
    WHEN 4 THEN 'Wednesday'
    WHEN 5 THEN 'Thursday'
    WHEN 6 THEN 'Friday'
    WHEN 7 THEN 'Saturday'
END
  AS start_day,
  CASE EXTRACT(DAYOFWEEK
  FROM
    pr_campaign.end_date)
    WHEN 1 THEN 'Sunday'
    WHEN 2 THEN 'Monday'
    WHEN 3 THEN 'Tuesday'
    WHEN 4 THEN 'Wednesday'
    WHEN 5 THEN 'Thursday'
    WHEN 6 THEN 'Friday'
    WHEN 7 THEN 'Saturday'
END
  AS end_day,
  root_sku.*
FROM
  `gum-eroski-dev.ETL.promotional_campaign` pr_campaign
INNER JOIN
  `gum-eroski-dev.ETL.promotional_to_sku` pr_to_sku
ON
  pr_to_sku.promo_id = pr_campaign.promo_id
  AND pr_to_sku.promo_year = pr_campaign.promo_year
INNER JOIN
  `gum-eroski-dev.ETL.root_sku` root_sku
ON
  root_sku.sku_root_id = pr_to_sku.sku_root_id;
  #create customer table
CREATE OR REPLACE TABLE
  `ETL.customer` AS
SELECT
  ID_CLIENTE_HUB AS customer_id,
IF
  (DESC_ESTATUS_TARJ_EROSKICLUB <> "ALTA PENDIENTE CONFIRMACION",
    "1",
    DESC_ESTATUS_TARJ_EROSKICLUB ) AS Eroski_club_loyalty_member,
IF
  (DESC_ESTATUS_SOCIO_ORO <> "ALTA PENDIENTE CONFIRMACION",
    "1",
    DESC_ESTATUS_SOCIO_ORO ) AS Eroski_gold_member,
IF
  (DESC_ESTATUS_TARJ_TRAVEL <> "ALTA PENDIENTE CONFIRMACION",
    "1",
    DESC_ESTATUS_TARJ_TRAVEL ) AS Eroski_travel_card_member,
  COD_SEXO AS gender,
IF
  (EXTRACT(DAYOFYEAR
    FROM
      CURRENT_DATE) < EXTRACT(DAYOFYEAR
    FROM
      PARSE_DATE("%d/%m/%Y",
        FECHADENACIMIENTO)),
    DATE_DIFF(CURRENT_DATE, PARSE_DATE("%d/%m/%Y",
        FECHADENACIMIENTO), YEAR) -1,
    DATE_DIFF(CURRENT_DATE, PARSE_DATE("%d/%m/%Y",
        FECHADENACIMIENTO), YEAR)) AS age,
  CP AS location
FROM
  `source_data.23_M_CLIENTES`;
