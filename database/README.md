# import data to PostgreSQL
We have 4 data sources.
* county
* sandag
* greatschool
* addresses_to_geocode.csv: dumped from tiger geocoding

## database server setup
* PostgreSQL 9.6 
* extensions installed
CREATE EXTENSION postgis;
CREATE EXTENSION fuzzystrmatch; --needed for postgis_tiger_geocoder
CREATE EXTENSION address_standardizer;
CREATE EXTENSION address_standardizer_data_us;
CREATE EXTENSION postgis_topology;
CREATE EXTENSION postgis_tiger_geocoder;

## local tools
* install postgresql client eg. sudo yum -y install postgresql96 
* install shp2pgsql. The tool is used to import shapefile for geographic data.

## load data
* county and addresses_to_geocode
Run load_county.sql in any Postgresql client(eg psql, pgadmin).
Noted, 'COPY' command supposes the data file resides on server rather than local host. If your host is different machine than server, run \copy command in psql instead.
* sandag
  * Run commands in load_sandag.bat. Be sure shp2pgsql is installed.
* greatschool
  * Run load_greatschool.sql to import school rating.
  * Run process_school.sql which generate school features for modeling.

## preprocess.sql
This script created various virtual/materialized views for data processing. Be sure to execute this before running python notebooks.

## materialize.sql
This script materialize certain views and created extra indices to speed up queries used by visualization and application.

## statistics.sql
This script includes commands for statistics like sampling, histogram etc

