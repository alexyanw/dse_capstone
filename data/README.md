### county
property and transaction from SD county
* PARDATA_TAB.txt - Property parcel records with pin(property identity number) as PK
* Address.txt - Property address, pin as PK
* MPR_Recent.txt - 2017 most recent appraisal by county.
* SalesData_AllYears_Distinct.txt - transaction with (pin, date) as PK
* YearBuiltAll.csv - year built information for each property, pin as PK
* foreclosure.csv(not used) - potential foreclosure transactions without price.

### addresses_to_geocode
The geolocation(lonlat) encoded from address for each property.

### sandag
GIS data downloaded from sandag
* SCHOOL - geographic boundary for each school: elem, middle, high, college, private in SD county
* Zip_Codes - geographic boundary for 113 zips in SD

### greatschool
2018 SD county school rating scraped from greatschool
