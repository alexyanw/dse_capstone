DROP TABLE IF EXISTS county_year_built;
CREATE TABLE county_year_built (
    pin varchar(10),
    effective_year integer,
    year_built integer,
);
COPY county_year_built FROM 'c:\wenyan\dse_capstone\data\county\YearBuiltAll.csv'
    WITH NULL '' DELIMITER E'\t' ENCODING 'utf-8'  CSV HEADER;

