DROP TABLE IF EXISTS greatschool_rating;
CREATE TABLE greatschool_rating (
    name varchar(256),
    address varchar(256),
    type varchar(64),
    grade varchar(32),
    rating smallint
);

COPY greatschool_rating FROM 'c:\wenyan\dse_capstone\data\school_rating.csv'
    WITH NULL '' DELIMITER ';' ENCODING 'utf-8'  CSV HEADER;

