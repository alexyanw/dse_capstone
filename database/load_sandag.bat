:: load from shape
:: If you see error "SRID (102646) in spatial_ref_sys", try converting shp to geojson and load from geojson(see section below)
::shp2pgsql -I -s 102646:4326 c:\wenyan\dse_capstone\data\sandag\SCHOOL\SCHOOL.shp sandag_schools | psql -U postgres -d sdra
::shp2pgsql -I -s 4326 c:\wenyan\dse_capstone\data\sandag\Zip_Codes\ZIP_CODES1.shp zip_codes |psql -U postgres -d sdra

:: load from geojson
cd c:\OSGeo4W64\bin
::ogr2ogr -f "PostgreSQL" PG:"dbname=sdra user=postgres" "c:\wenyan\dse_capstone\data\sandag\schools.geojson" -nln sandag_schools -append -t_srs EPSG:4326 
ogr2ogr -f "PostgreSQL" PG:"dbname=sdra user=postgres" "c:\wenyan\dse_capstone\data\sandag\zip_codes.geojson" -nln zip_codes -append
