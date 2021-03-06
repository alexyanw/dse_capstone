CREATE TABLE IF NOT EXISTS county_transactions (
    code varchar(1),
    pin varchar(10),
    trailer_pin varchar(10),
    doc_date date,
    doc_num integer,
    price integer,
    address varchar(128)
);

COPY county_transactions FROM 'c:\wenyan\dse_capstone\data\county\SalesData_AllYears_Distinct.txt'
    WITH NULL '' DELIMITER E'\t' ENCODING 'utf-8'  CSV HEADER;

create index county_transaction_pin on county_transactions(pin);

CREATE TABLE IF NOT EXISTS county_addresses (
    pin varchar(10),
    str_no varchar(16),
    street varchar(128),
    st_type varchar(16),
    unit_no varchar(128),
    city varchar(32),
    state varchar(8),
    zip varchar(10)
);
COPY county_addresses FROM 'c:\wenyan\dse_capstone\data\county\Address.txt'
    WITH NULL '' DELIMITER E'\t' ENCODING 'utf-8'  CSV HEADER;
create index county_addresses_pin on county_addresses(pin);

CREATE TABLE IF NOT EXISTS county_mpr (
    pin varchar(10),
    str_no varchar(16),
    street varchar(128),
    land integer,
    improvement integer
);

COPY county_mpr FROM 'c:\wenyan\dse_capstone\data\county\MPR_Recent.txt'
    WITH NULL '' DELIMITER E'\t' ENCODING 'utf-8'  CSV HEADER;

CREATE TABLE IF NOT EXISTS county_properties (
    par_status                            varchar(1),
    par_parcel_number                     varchar(10),
    par_tax_rate_area                     varchar(5),
    par_land_use_code                     integer,
    par_report_group                      varchar(3),
    par_neighborhood                      varchar(2),
    par_appraiser_id                      varchar(5),
    par_act_date                          integer,
    par_qual_class_shape                  varchar(5),
    par_year_effective                    integer,
    par_total_lvg_area                    integer,
    par_bedrooms                          integer,
    par_bathroom                          integer,
    par_addition_area                     integer,
    par_garage_conversion                 varchar(1),
    par_garage_stalls                     varchar(3),
    par_carport_stalls                    varchar(3),
    par_pool                              varchar(1),
    par_view                              varchar(1),
    par_usable_sq_feet                    integer,
    par_sheet_status                      varchar(1),
    par_sheet_date                        varchar(6),
    par_cut_code                          varchar(1),
    par_sales_code                        varchar(1),
    permit_code                           varchar(1),
    par_acreage                           integer,
    par_units                             integer,
    par_current_land                      integer,
    par_current_imps                      integer,
    par_fixtures                          integer,
    par_personal_property                 integer,
    par_exem_code1                        varchar(1),
    par_exem_amount1                      integer,
    par_exem_code2                        varchar(1),
    par_exem_amount2                      integer,
    par_exem_code3                        varchar(1),
    par_exem_amount3                      integer,
    par_value_change_code                 varchar(1),
    par_value_action_date                 varchar(6),
    par_value_action_code                 varchar(1),
    par_value_change_date                 varchar(6),
    par_tax_status                        varchar(1),
    par_construction_exempt               varchar(1),
    par_base_land_values                       integer,
    par_base_improvement_values                integer,
    par_base_value_change_code                 varchar(1),
    par_base_value_active_date                 varchar(6),
    par_base_value_active_code                 varchar(1),
    par_prior_land1_values                integer,
    par_prior_improvement1_values         integer,
    par_prior_fixtures1_values            integer,
    par_prior_personal_property1_values   integer,
    par_roll_exem_code1                   varchar(1),
    par_roll_exem_amount1                 integer,
    par_roll_exem_code2                   varchar(1),
    par_roll_exem_amount2                 integer,
    par_roll_exem_code3                   varchar(1),
    par_roll_exem_amount3                 integer,
    par_prior_value_change_code1          varchar(1),
    par_prior_tax_roll_yr1                varchar(2),
    par_prior_land2_values                integer,
    par_prior_improvement2_values         integer,
    par_prior_fixture2_values             integer,
    par_prior_personal_property2_values   integer,
    par_exem_code1_2                      varchar(1),
    par_exem_amount1_2                    integer,
    par_exem_code2_2                      varchar(1),
    par_exem_amount2_2                    integer,
    par_exem_code3_2                      varchar(1),
    par_exem_amt3_2                       integer,
    par_prior_value_change_code2          varchar(1),
    par_prior_tax_roll_year2              varchar(2),
    par_prior_land_values3                integer,
    par_prior_improvement_values3         integer,
    par_prior_fixture_values3             integer,
    par_prior_personal_property_values3   integer,
    par_exem_code1_3                      varchar(1),
    par_exem_amount1_3                    integer,
    par_exem_code2_3                      varchar(1),
    par_exem_amt2_3                       integer,
    par_exem_code3_3                      varchar(1),
    par_exem_amount3_3                    integer,
    par_prior_value_change_code3          varchar(1),
    par_prior_tax_roll_year_3             varchar(2),
    par_official_document_number1         varchar(6),
    par_official_document_date1           varchar(6),
    par_official_mult_cd1                 varchar(1),
    par_official_stamp1                   integer,
    par_official_price1                   integer,
    par_official_fp_code1                 varchar(1),
    par_official_frt_int1                 varchar(3),
    par_official_supplemental_date1       varchar(6),
    par_official_supplemental_code1       varchar(1),
    par_official_prt_code1                varchar(1),
    par_official_document_number2         varchar(6),
    par_official_document_date2           varchar(6),
    par_official_mult_cd2                 varchar(1),
    par_official_stamp2                   integer,
    par_official_price2                   integer,
    par_official_fp_code2                 varchar(1),
    par_official_frt_int2                 varchar(3),
    par_official_supplemental_date2       varchar(6),
    par_official_supplemental_code2       varchar(1),
    par_official_prt_code2                varchar(1),
    par_official_document_number3         varchar(6),
    par_official_document_date3           varchar(6),
    par_official_mult_cd3                 varchar(1),
    par_official_stamp3                   integer,
    par_official_price3                   integer,
    par_official_fp_code3                 varchar(1),
    par_official_frt_int3                 varchar(3),
    par_official_supplemental_date3       varchar(6),
    par_official_supplemental_code3       varchar(1),
    par_official_prt_code3                varchar(1),
    par_official_document_number4         varchar(6),
    par_official_document_date4           varchar(6),
    par_official_mult_cd4                 varchar(1),
    par_official_stamp4                   integer,
    par_official_price4                   integer,
    par_official_fp_code4                 varchar(1),
    par_official_frt_int4                 varchar(3),
    par_official_supplemental_date4       varchar(6),
    par_official_supplemental_code4       varchar(1),
    par_official_prt_code4                varchar(1),
    par_official_document_number5         varchar(6),
    par_official_document_date5           varchar(6),
    par_official_mult_cd5                 varchar(1),
    par_official_stamp5                   integer,
    par_official_price5                   integer,
    par_official_fp_code5                 varchar(1),
    par_official_frt_int5                 varchar(3),
    par_official_supplemental_date5       varchar(6),
    par_official_supplemental_code5       varchar(1),
    par_official_prt_code5                varchar(1),
    par_official_document_number6         varchar(6),
    par_official_document_date6           varchar(6),
    par_official_mult_cd6                 varchar(1),
    par_official_stamp6                   integer,
    par_official_price6                   integer,
    par_official_fp_code6                 varchar(1),
    par_official_frt_int6                 varchar(3),
    par_official_supplemental_date6       varchar(6),
    par_official_supplemental_code6       varchar(1),
    par_official_prt_code6                varchar(1),
    par_ncp_permit_number                 varchar(10),
    par_ncp_permit_date                   varchar(6),
    par_ncp_bc_code                       varchar(3),
    par_ncp_permit_value                  integer,
    par_ncp_final_insp_date               varchar(6),
    par_ncp_permit_status                 varchar(1),
    par_map_permit_number                 varchar(10),
    par_map_permit_date                   varchar(6),
    par_map_bc_code                       varchar(3),
    par_map_permit_value                  integer,
    par_map_final_insp_date               varchar(6),
    par_map_permit_status                 varchar(1),
    par_bpi_permit_number1                varchar(10),
    par_bpi_permit_date1                  varchar(6),
    par_bpi_bc_code1                      varchar(3),
    par_bpi_permit_value1                 integer,
    par_bpi_final_insp_date1              varchar(6),
    par_bpi_permit_status1                varchar(1),
    par_bpi_permit_number2                varchar(10),
    par_bpi_permit_date2                  varchar(6),
    par_bpi_bc_code2                      varchar(3),
    par_bpi_permit_value2                 integer,
    par_bpi_final_insp_date2              varchar(6),
    par_bpi_permit_status2                varchar(1),
    par_bpi_permit_number3                varchar(10),
    par_bpi_permit_date3                  varchar(6),
    par_bpi_bc_code3                      varchar(3),
    par_bpi_permit_value3                 integer,
    par_bpi_final_insp_date3              varchar(6),
    par_bpi_permit_status3                varchar(1),
    par_bpi_permit_number4                varchar(10),
    par_bpi_permit_date4                  varchar(6),
    par_bpi_bc_code4                      varchar(3),
    par_bpi_permit_value4                 integer,
    par_bpi_final_insp_date4              varchar(6),
    par_bpi_permit_status4                varchar(1),
    par_additions_permit                  varchar(1),
    par_permit_supplemental_date          varchar(6),
    par_permit_supplemental_code          varchar(1),
    par_end_record_flag_z                 varchar(1)
);

COPY county_properties FROM 'c:\wenyan\dse_capstone\data\county\PARDATA_TAB.txt'
    WITH NULL '' DELIMITER E'\t' ENCODING 'utf-8'  CSV HEADER;
;

create index county_property_pin on county_properties(par_parcel_number);

CREATE TABLE IF NOT EXISTS county_year_built (
    pin varchar(10),
    effective_year integer,
    year_built integer
);
COPY county_year_built FROM 'c:\wenyan\dse_capstone\data\county\YearBuiltALL.csv'
WITH NULL '' DELIMITER E'\t' ENCODING 'utf-8'  CSV HEADER;

CREATE TABLE IF NOT EXISTS county_foreclosures(
    inst_number varchar(12),
    date date,
    doctype smallint,
    doctypedesc varchar(32),
    pin	varchar(10),
    ZIP varchar(5)
);
COPY county_foreclosure FROM 'c:\wenyan\dse_capstone\data\county\foreclosure.csv'
    WITH NULL '' DELIMITER ',' ENCODING 'utf-8'  CSV HEADER;
;

CREATE TABLE IF NOT EXISTS addresses_to_geocode(
    addid serial PRIMARY KEY, 
    address text,
    lon numeric, 
    lat numeric, 
    streetno text,
    streetname text,
    streettype text,
    city text,
    state text,
    zip text,
    new_address text,
    rating integer,
    pin varchar(10),
    status integer
);

COPY addresses_to_geocode FROM 'c:\wenyan\dse_capstone\data\addresses_to_geocode.csv'
    WITH NULL '' DELIMITER E'\t' ENCODING 'utf-8'  CSV HEADER;
;
create index addresses_to_geocode_pin on addresses_to_geocode(pin);


