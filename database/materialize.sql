CREATE MATERIALIZED VIEW property_transaction_valid AS
SELECT * 
FROM property_address_transactions
WHERE sold_price > 0 AND sold_price < 3000000
  AND sqft_price > 0 AND sqft_price < 2000
  AND sqft < 10000
  AND num_bed < 10 AND num_bath < 10




