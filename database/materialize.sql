CREATE MATERIALIZED VIEW property_transaction_valid AS
SELECT * 
FROM property_address_transactions
WHERE sold_price > 0 AND sold_price < 3000000
  AND sqft_price > 0 AND sqft_price < 2000
  AND sqft < 10000
  AND num_bed < 10 AND num_bath < 10
  AND lon is not null


CREATE MATERIALIZED VIEW history_statistics AS
WITH valid_transactions as (
SELECT *,
  EXTRACT(YEAR FROM date) as year, EXTRACT(MONTH FROM date) as month
FROM property_address_school_transactions
WHERE sold_price > 0 AND sold_price < 8000000
  AND sqft_price > 0 AND sqft_price < 4000
  AND sqft < 100000
  AND num_bed < 10 AND num_bath < 10
  AND lon is not null
),
transaction_tiles as (
SELECT *,
    ntile(3) OVER (PARTITION BY year ORDER BY sqft_price) AS tile
FROM valid_transactions
)
-- yearly
SELECT zip, year, 0 as month, 0 as tile, count(1) as volume, AVG(sqft_price)::decimal(10,1) as avg_sqft_price, median(sqft_price)::decimal(10,1) as median_sqft_price, median(sold_price)::decimal(10,1) as median_sold_price, year - avg(year_built)::integer as avg_sold_age, avg(sqft)::decimal(10,1) as avg_sold_sqft
FROM transaction_tiles
GROUP BY zip, year
UNION
SELECT '' AS zip, year, 0 as month, 0 as tile, count(1) as volume, AVG(sqft_price)::decimal(10,1) as avg_sqft_price, median(sqft_price)::decimal(10,1) as median_sqft_price, median(sold_price)::decimal(10,1) as median_sold_price, year - avg(year_built)::integer as avg_sold_age, avg(sqft)::decimal(10,1) as avg_sold_sqft
FROM transaction_tiles
GROUP BY year
UNION
-- yearly, tile
SELECT zip, year, 0 as month, tile, count(1) as volume, AVG(sqft_price::decimal(10,1)) as avg_sqft_price, median(sqft_price)::decimal(10,1) as median_sqft_price, median(sold_price)::decimal(10,1) as median_sold_price, year - avg(year_built)::integer as avg_sold_age, avg(sqft)::decimal(10,1) as avg_sold_sqft
FROM transaction_tiles
GROUP BY zip, year, tile
UNION
SELECT '' AS zip, year, 0 as month, tile, count(1) as volume, AVG(sqft_price)::decimal(10,1) as avg_sqft_price, median(sqft_price)::decimal(10,1) as median_sqft_price, median(sold_price)::decimal(10,1) as median_sold_price, year - avg(year_built)::integer as avg_sold_age, avg(sqft)::decimal(10,1) as avg_sold_sqft
FROM transaction_tiles
GROUP BY year, tile
UNION
-- monthly
SELECT zip, year, month, 0 as tile, count(1) as volume, AVG(sqft_price)::decimal(10,1) as avg_sqft_price, median(sqft_price)::decimal(10,1) as median_sqft_price, median(sold_price)::decimal(10,1) as median_sold_price, year - avg(year_built)::integer as avg_sold_age, avg(sqft)::decimal(10,1) as avg_sold_sqft
FROM transaction_tiles
GROUP BY zip, year, month
UNION
SELECT '' AS zip, year, month, 0 as tile, count(1) as volume, AVG(sqft_price)::decimal(10,1) as avg_sqft_price, median(sqft_price)::decimal(10,1) as median_sqft_price, median(sold_price)::decimal(10,1) as median_sold_price, year - avg(year_built)::integer as avg_sold_age, avg(sqft)::decimal(10,1) as avg_sold_sqft
FROM transaction_tiles
GROUP BY year, month


-- school stats
CREATE MATERIALIZED VIEW school_statistics AS
WITH dates as
(SELECT generate_series(date '1983-12-31', date '2017-12-31', '1 year') as date
),
school_year as(
select s.*, extract(year from d.date) as year
FROM schools s, dates d
where s.opendate <= d.date
)
SELECT o.year, o.zip, o.count as school_count, o.avg_rating as avg_school_rating, e.elementary_count, e.avg_elementary_rating, m.middle_count, m.avg_middle_rating, h.high_count, h.avg_high_rating
FROM
(SELECT year, zip, count(1), avg(rating)::decimal(10, 1) as avg_rating
FROM school_year
GROUP BY year,zip) o
FULL OUTER JOIN
(SELECT year, zip, count(1) as elementary_count, avg(rating)::decimal(10, 1) as avg_elementary_rating
FROM school_year
WHERE elementary = true
GROUP BY year,zip) e
ON o.zip = e.zip and o.year = e.year
FULL OUTER JOIN
(SELECT year, zip, count(1) as middle_count, avg(rating)::decimal(10, 1) as avg_middle_rating
FROM school_year
WHERE middle = true
GROUP BY year,zip) m
ON o.zip=m.zip and o.year=m.year
FULL OUTER JOIN
(SELECT year, zip, count(1) as high_count, avg(rating)::decimal(10, 1) as avg_high_rating
FROM school_year
WHERE high = true
GROUP BY year,zip) h
ON o.zip=h.zip and o.year = h.year 

