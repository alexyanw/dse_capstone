--  In case importing shapefile with shp2pgsql not working and ogr2ogr can't be run on database server, below is the workaround way to load sandag data.
SET CLIENT_ENCODING TO UTF8;
SET STANDARD_CONFORMING_STRINGS TO ON;
BEGIN;
CREATE TABLE "sandag_schools" (gid serial,
"cdscode" varchar(14),
"district" varchar(90),
"school" varchar(90),
"street" varchar(201),
"city" varchar(25),
"zip" numeric,
"opendate" date,
"charter" varchar(1),
"doctype" varchar(50),
"soctype" varchar(50),
"gsoffered" varchar(8),
"shortname" varchar(20),
"id" numeric,
"priv" varchar(1));
ALTER TABLE "sandag_schools" ADD PRIMARY KEY (gid);
SELECT AddGeometryColumn('','sandag_schools','geom','4326','POINT',2);
COPY sandag_schools FROM 'c:\wenyan\dse_capstone\data\sandag\sandag_schools.csv' WITH NULL '' DELIMITER ';' ENCODING 'utf-8'  CSV HEADER;
CREATE INDEX ON "sandag_schools" USING GIST ("geom");
COMMIT;
ANALYZE "sandag_schools";


SET CLIENT_ENCODING TO UTF8;
SET STANDARD_CONFORMING_STRINGS TO ON;
BEGIN;
CREATE TABLE "zip_codes" (gid serial,
"zip" int8,
"community" varchar(20),
"shape_star" numeric,
"shape_stle" numeric);
ALTER TABLE "zip_codes" ADD PRIMARY KEY (gid);
SELECT AddGeometryColumn('','zip_codes','wkb_geometry','4326','MULTIPOLYGON',2);
COPY zip_codes FROM 'c:\wenyan\dse_capstone\data\\SalesData_AllYears_Distinct.txt' WITH NULL '' DELIMITER ';' ENCODING 'utf-8'  CSV HEADER;

CREATE INDEX ON "zip_codes" USING GIST ("wkb_geometry");
COMMIT;
ANALYZE "zip_codes";
