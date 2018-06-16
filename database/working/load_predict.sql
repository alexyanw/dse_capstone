CREATE TABLE IF NOT EXISTS property_estimations (
    id integer,
    pin varchar(10),
    date date,
    estimation float
);

COPY property_estimations FROM 'c:\wenyan\dse_capstone\data\dump\predict_full_4.csv' WITH NULL '' DELIMITER ',' ENCODING 'utf-8'  CSV HEADER;

create index property_estimate_pin on property_estimations(pin);

CREATE TABLE IF NOT EXISTS residual_stds (
    id integer,
    date date,
    std float
);

COPY residual_stds FROM 'c:\wenyan\dse_capstone\data\dump\residual_std.csv' WITH NULL '' DELIMITER ',' ENCODING 'utf-8'  CSV HEADER;

