-- run once
CREATE INDEX property_addresses_lonlat ON property_addresses(ST_GeomFromText('POINT(' || lon || ' ' || lat || ')', 4326));

ALTER TABLE sandag_schools  ALTER COLUMN id TYPE integer;
ALTER TABLE sandag_schools  ALTER COLUMN zip TYPE integer; 

update sandag_schools set school = 'Perkins Elementary' where school = 'Perkins K-8';
update sandag_schools set school = 'Mount Vernon Elementary' where school = 'Mt. Vernon Elementary';
update sandag_schools set street = '3510 Newton Avenu' where school = 'Emerson/Bandini Elementary;
-- run once done

CREATE MATERIALIZED VIEW IF NOT EXISTS schools AS
WITH school_grade AS (
  SELECT *,substring(gsoffered from '^[^-]*') as grade_start, substring(gsoffered from '[^-]*$') as grade_end
  FROM sandag_schools
),
school_grade_int AS (
  SELECT id,school,district,street,city,zip,opendate,doctype,soctype,gsoffered,wkb_geometry,
	  REPLACE(REPLACE(grade_start, 'P', '-1'), 'K', '0')::integer AS grade_start, 
	  REPLACE(REPLACE(grade_end, 'P', '-1'), 'K', '0')::integer AS grade_end
  FROM school_grade
),
school_grade_bool AS (
  SELECT *,
    CASE
  	WHEN (grade_start<=1 and grade_end >=1) or (grade_start>1 and grade_start <=5) THEN true
  	ELSE false
    END AS elementary,
    CASE
  	WHEN (grade_start<=7 and grade_end >=7) or (grade_start>8 and grade_start <=8) THEN true
  	ELSE false
    END AS middle,
    CASE
  	WHEN (grade_start<=10 and grade_end >=10) or (grade_start>12 and grade_start <=12) THEN true
  	ELSE false
    END AS high,
    substring(school from '([^,]+)') as name,
    substring(street from '([^,]+)') as street_address
  FROM school_grade_int
),
greatschool_address AS(
    SELECT substring(g.address from '([^,]+)') as street_address, *
    FROM greatschool_rating g
    WHERE type != 'Private'
)
SELECT s.*, g.rating
FROM school_grade_bool s LEFT OUTER JOIN greatschool_address g
ON (levenshtein(lower(s.street_address),lower(g.street_address))<=5 
    OR s.street_address ILIKE '%' || g.street_address || '%'
    OR g.street_address ILIKE '%' || s.street_address || '%')
AND (g.name ILIKE '%' || s.name || '%'
    OR s.name ILIKE '%' || g.name || '%'
    )
;

CREATE VIEW school_feature AS
with school_elem_rating_zip_avg AS (
  select zip,avg(rating) as rating from schools
  where soctype != 'Private' and rating is not null and elementary=true
  group by zip
),
school_middle_rating_zip_avg AS (
  select zip,avg(rating) as rating from schools
  where soctype != 'Private' and rating is not null and middle=true
  group by zip
),
school_high_rating_zip_avg AS (
  select zip,avg(rating) as rating from schools
  where soctype != 'Private' and rating is not null and high=true
  group by zip
)
SELECT s.*,
  CASE
    WHEN s.rating is null AND s.elementary=true THEN e.rating
    WHEN s.rating is null AND s.middle=true THEN m.rating
    WHEN s.rating is null AND s.high=true THEN h.rating
    ELSE s.rating
  END AS rating_valid
from schools s
  LEFT OUTER JOIN school_elem_rating_zip_avg e ON s.zip=e.zip
  LEFT OUTER JOIN school_middle_rating_zip_avg m ON s.zip=m.zip
  LEFT OUTER JOIN school_high_rating_zip_avg h ON s.zip=h.zip
WHERE s.soctype != 'Private'
;

CREATE MATERIALIZED VIEW IF NOT EXISTS property_school_distance AS
SELECT p.pin, s.id as school_id, ST_Distance_Sphere(ST_GeomFromText('POINT(' || p.lon || ' ' || p.lat || ')', 4326), s.wkb_geometry) AS distance
FROM property_addresses p, school_feature s
WHERE p.zip = s.zip::text
  AND p.lon is not null
;

CREATE materialized VIEW IF NOT EXISTS property_school_elementary AS
WITH property_elementary_distance AS (
    SELECT d.pin,d.school_id,d.distance,s.rating_valid
    FROM property_school_distance d, school_feature s
    WHERE d.school_id=s.id AND s.elementary=true
)
SELECT p.pin, p.school_id, p.distance, p.rating_valid as rating, p.rnum
FROM (SELECT *,ROW_NUMBER() OVER (partition BY pin ORDER BY distance) AS rnum
      FROM property_elementary_distance) p
WHERE p.rnum < 4
;

CREATE materialized VIEW IF NOT EXISTS property_school_middle AS
WITH property_middle_distance AS (
    SELECT d.pin,d.school_id,d.distance,s.rating_valid
    FROM property_school_distance d, school_feature s
    WHERE d.school_id=s.id AND s.middle=true
)
SELECT p.pin, p.school_id, p.distance, p.rating_valid as rating, p.rnum
FROM (SELECT *,ROW_NUMBER() OVER (partition BY pin ORDER BY distance) AS rnum
      FROM property_middle_distance) p
WHERE p.rnum < 4
;

CREATE materialized VIEW IF NOT EXISTS property_school_high AS
WITH property_high_distance AS (
    SELECT d.pin,d.school_id,d.distance, s.rating_valid
    FROM property_school_distance d, school_feature s
    WHERE d.school_id=s.id AND s.high=true
)
SELECT p.pin, p.school_id, p.distance, p.rating_valid as rating, p.rnum
FROM (SELECT *,ROW_NUMBER() OVER (partition BY pin ORDER BY distance) AS rnum
      FROM property_high_distance) p
WHERE p.rnum < 4
;

CREATE MATERIALIZED VIEW IF NOT EXISTS property_closest_schools AS
SELECT 
  CASE
    WHEN e.pin is not null THEN e.pin
    WHEN m.pin is not null THEN m.pin
    ELSE h.pin
  END as pin, e.school_id as elem_id, m.school_id as middle_id, h.school_id as high_id,
  e.rating as elem_rating, e.distance as min_elem_distance,
  m.rating as middle_rating, m.distance as min_middle_distance,
  h.rating as high_rating, h.distance as min_high_distance
FROM property_school_elementary e
    FULL OUTER JOIN property_school_middle m ON e.pin=m.pin
    FULL OUTER JOIN property_school_high h ON e.pin=h.pin
WHERE (e.rnum=1 or e.rnum is null) AND (m.rnum=1 or m.rnum is null) AND (h.rnum=1 or h.rnum is null)
;

CREATE MATERIALIZED View IF NOT EXISTS property_address_schools AS
WITH property_elementary AS (
    select pin, avg(rating) as avg_rating, avg(distance) as avg_distance
    from property_school_elementary
    group by pin
),
property_middle AS (
    select pin, avg(rating) as avg_rating, avg(distance) as avg_distance
    from property_school_middle
    group by pin
),
property_high AS (
    select pin, avg(rating) as avg_rating, avg(distance) as avg_distance
    from property_school_high
    group by pin
)
SELECT pa.*, 
  pc.elem_rating, pc.middle_rating, pc.high_rating, 
  pc.min_elem_distance, pc.min_middle_distance, pc.min_high_distance, 
  pe.avg_distance AS avg_elem_distance, pe.avg_rating AS avg_elem_rating, 
  pm.avg_distance AS avg_middle_distance, pm.avg_rating AS avg_middle_rating,
  ph.avg_distance AS avg_high_distance, ph.avg_rating AS avg_high_rating
FROM property_addresses pa 
  LEFT OUTER JOIN property_closest_schools pc ON pa.pin=pc.pin
  LEFT OUTER JOIN property_elementary pe ON pa.pin=pe.pin
  LEFT OUTER JOIN property_middle pm ON pa.pin=pm.pin
  LEFT OUTER JOIN property_high ph ON pa.pin=ph.pin
;

CREATE MATERIALIZED VIEW IF NOT EXISTS property_address_school_transactions AS
SELECT
    p.*,
    t.sold_price, t.date,
    t.sold_price / p.sqft AS sqft_price
FROM property_address_schools p, transactions t
WHERE p.pin = t.pin
ORDER by t.date
;
