drop materialized view if exists redfin_prop_multiple CASCADE;
CREATE materialized view redfin_prop_multiple AS
SELECT DISTINCT t.redfin_id
   FROM ( SELECT redfin_transactions.redfin_id,
           row_number() OVER (PARTITION BY redfin_transactions.redfin_id) AS rnum
           FROM redfin_transactions
           WHERE redfin_transactions.source like '%public%') t
  WHERE t.rnum > 1;

drop materialized view if exists redfin_foreclosure;
CREATE materialized view redfin_foreclosure AS
 WITH public_view AS (
         SELECT redfin_transactions.redfin_id,
            redfin_transactions.date,
            redfin_transactions.event,
            redfin_transactions.price,
            redfin_transactions.appreciation,
            redfin_transactions.source
           FROM redfin_transactions
          WHERE redfin_transactions.source::text ~~ '%Public%'::text
        ), mls_view AS (
         SELECT redfin_transactions.redfin_id,
            redfin_transactions.date,
            redfin_transactions.event,
            redfin_transactions.price,
            redfin_transactions.appreciation,
            redfin_transactions.source
           FROM redfin_transactions
          WHERE redfin_transactions.source::text ~~ '%MLS%'::text
        )
 SELECT rp.number, rp.street, rp.city, rp.zip,
    p.redfin_id, p.date, p.event, p.price, p.source
   FROM public_view p,
    redfin_properties rp,
    redfin_prop_multiple rpm
  WHERE p.date < '2017-06-01'::date AND p.redfin_id = rp.redfin_id AND p.redfin_id = rpm.redfin_id AND NOT (EXISTS ( SELECT 1
           FROM mls_view m
          WHERE m.redfin_id = p.redfin_id AND ((p.date - m.date) >= '-5'::integer AND (p.date - m.date) <= 5 OR (p.price - m.price) >= '-10000'::integer AND (p.price - m.price) <= 10000)))
  ORDER BY p.date DESC;

create view redfin_public_transactions AS
 SELECT rp.number, rp.street, rp.city, rp.zip,
    rt.redfin_id, rt.date, rt.price
   FROM redfin_transactions rt,
    redfin_properties rp
  WHERE rt.redfin_id = rp.redfin_id AND rt.source::text ~~ '%Public%'::text AND rt.event::text = 'sold'::text;

create materialized view county_redfin_mapping AS
 SELECT pa.pin,
    rf.redfin_id,
    upper(concat(pa.str_no, ' ', pa.street, ' ', pa.st_type, ', ', pa.zip)) AS county_addr,
    upper(concat(rf.number, ' ', rf.street, ', ', rf.zip)) AS redfin_addr,
    levenshtein(upper(concat(pa.street, ' ', pa.st_type)), upper(rf.street::text)) AS distance,
    pa.date AS date1,
    rf.date AS date2
   FROM property_transaction_valid pa,
    redfin_public_transactions rf
  WHERE pa.str_no::text = rf.number::text AND pa.zip = rf.zip::text AND levenshtein(upper(concat(pa.street, ' ', pa.st_type)), upper(rf.street::text)) <= 5 AND (pa.date - rf.date) >= '-5'::integer AND (pa.date - rf.date) <= 5 AND (pa.sold_price - rf.price) >= '-5000'::integer AND (pa.sold_price - rf.price) <= 5000 AND pa.date > '1988-01-01'::date AND rf.date > '1988-01-01'::date;
