CREATE TABLE addresses_to_geocode(
    addid serial PRIMARY KEY, 
    pin varchar(10),
    address text,
    new_address text,
    lon numeric, 
    lat numeric, 
    streetno text,
    streetname text,
    streettype text,
    city text,
    state text,
    zip text,
    rating integer,
    status integer
);

INSERT INTO addresses_to_geocode(pin, address) 
select pin, concat(str_no, ' ', street, ' ', st_type, ', ', city, ', ', state, ' ', substring(zip,1,5)) 
from county_addresses 
where str_no is not null and str_no != '0' and street is not null
-- select par_parcel_number, concat(str_no, ' ', substring(str_add, 1, length(str_add)-2), ', CA') FROM county_properties 
;

UPDATE addresses_to_geocode a
SET status=1
FROM property_address_transactions p
WHERE p.pin = a.pin AND
  p.sold_price > 0 AND p.sqft < 10000
;

CREATE INDEX addresses_to_geocode_pin on addresses_to_geocode(pin);

UPDATE addresses_to_geocode
  SET  (rating, new_address, lon, lat, streetno, streetname, streettype, city, state,  zip)
    = ( COALESCE(g.rating,-1), pprint_addy(g.addy),
       ST_X(g.geomout)::numeric(8,5), ST_Y(g.geomout)::numeric(8,5), (g.addy).address, (g.addy).streetname, (g.addy).streettypeabbrev, (g.addy).location, (g.addy).stateabbrev, (g.addy).zip)
FROM (SELECT addid, address
      FROM addresses_to_geocode
      WHERE rating IS NULL AND status = 1
      ORDER BY addid LIMIT 30000) As a
    LEFT JOIN LATERAL geocode(a.address,1) As g ON true
WHERE a.addid = addresses_to_geocode.addid;


-- using standardize_address from Pagc_Normalize_Address
UPDATE addresses_to_geocode
  SET  (rating, new_address, lon, lat, streetno, streetname, streettype, city, state,  zip)
    = ( COALESCE(g.rating,-1), pprint_addy(g.addy),
       ST_X(g.geomout)::numeric(8,5), ST_Y(g.geomout)::numeric(8,5), (g.addy).address, (g.addy).streetname, (g.addy).streettypeabbrev, (g.addy).location, (g.addy).stateabbrev, (g.addy).zip)
FROM (
    SELECT addid, standardize_address('tiger.pagc_lex', 'tiger.pagc_gaz', 'tiger.pagc_rules', address) As sa
    FROM addresses_to_geocode
    WHERE rating IS NULL AND status = 1
    ORDER BY addid LIMIT 30000
) As a
LEFT JOIN LATERAL geocode(ROW(REGEXP_REPLACE((a.sa).house_num, '[^0-9]+', ''), (a.sa).predir, (a.sa).name,(a.sa).suftype, (a.sa).sufdir, (a.sa).unit , (a.sa).city, (a.sa).state, (a.sa).postcode, true)::norm_addy,1) As g ON true
WHERE a.addid = addresses_to_geocode.addid;
