-- drop duplicates; transform
DROP VIEW IF EXISTS properties CASCADE;
CREATE VIEW properties AS
SELECT
    par_parcel_number,
    par_land_use_code/100 as land_use_code,
    par_land_use_code%100 as land_use_subcode,
	CASE
		WHEN par_year_effective <= 17 THEN par_year_effective + 2000
		ELSE par_year_effective + 1900
	END as par_year_effective,
    par_total_lvg_area,
    par_acreage,
    par_bedrooms,
    par_bathroom,
    par_addition_area,
    par_garage_conversion,
    par_pool,
    par_view,
    par_usable_sq_feet,
    par_units,
    par_current_land,
    par_current_imps,
    par_personal_property,
    par_tax_status
FROM county_properties
;

-- cleaning: excluding invalid, outlier
DROP VIEW IF EXISTS property_features CASCADE;
CREATE VIEW property_features AS
SELECT 
    p.par_parcel_number AS pin,
    p.land_use_subcode,
	y.year_built,
    p.par_total_lvg_area AS sqft,
    p.par_usable_sq_feet AS usable_sqft,
    p.par_acreage AS acre,
    p.par_bedrooms AS num_bed,
    p.par_bathroom/10 AS num_bath,
    p.par_pool AS pool,
    p.par_view AS view,
    p.par_current_land AS eval_land,
    p.par_current_imps AS eval_imps
FROM properties p, county_year_built y
WHERE p.par_parcel_number = y.pin AND
  p.land_use_code = 1 AND p.land_use_subcode >= 11 AND p.land_use_subcode <= 17 
  AND p.par_total_lvg_area > 0
;

DROP VIEW IF EXISTS transactions;
CREATE VIEW transactions AS
SELECT
    pin,
    code,
    doc_date date,
    price sold_price
FROM county_transactions
WHERE code != 'M'
;

DROP VIEW IF EXISTS addresses CASCADE;
CREATE VIEW addresses AS
SELECT
    a.pin,
    a.str_no,
    a.street,
    a.st_type,
    a.unit_no,
    a.city,
    substring(a.zip, 1,5) zip
FROM county_addresses a
;

DROP VIEW IF EXISTS property_addresses;
CREATE VIEW property_addresses AS
SELECT
    f.*,
    a.str_no,
    a.street,
    a.st_type,
    a.unit_no,
    a.city,
    a.zip
FROM addresses a, property_features f
WHERE a.pin = f.pin
;

DROP VIEW IF EXISTS property_transactions;
CREATE VIEW property_transactions AS
SELECT
    p.*,
    t.sold_price, t.date,
    t.sold_price / p.sqft AS sqft_price
FROM property_features p, transactions t
WHERE p.pin = t.pin
ORDER by t.date DESC
;

DROP VIEW IF EXISTS property_address_transactions;
CREATE VIEW property_address_transactions AS
SELECT
    pa.*,
    t.sold_price, t.date,
    t.sold_price / pa.sqft AS sqft_price
FROM property_addresses pa, transactions t
WHERE pa.pin = t.pin
ORDER by t.date DESC
;

