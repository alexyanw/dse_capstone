:: load from shape, problem in geom
::shp2pgsql -I -s 4326 c:\wenyan\dse_capstone\data\sandag\SCHOOL\SCHOOL.shp sandag_schools | psql -U postgres -d sdra
:: load from geojson
cd c:\OSGeo4W64\bin
ogr2ogr -f "PostgreSQL" PG:"dbname=sdra user=postgres" "c:\projects\sdra\public\js\schools.geojson" -nln sandag_schools -append -t_srs EPSG:4326 
::ogr2ogr -f "PostgreSQL" PG:"dbname=sdra user=postgres" "c:\projects\sdra\public\js\zips.geojson" -nln sandag_zips -append
