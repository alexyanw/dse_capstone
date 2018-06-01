# dse_capstone - San Diego County Housing Market Analysis
This repository is used for source code development for analytical tasks. Visualization and prototype project is split to another project.

## python version and modules
* Anaconda 3.6.4 is used.
* Python modules installed
  * conda install sqlalchemy psycopg2
  * conda install -c anaconda seaborn
  * pip install python-dotenv
  * pip install bayesian-optimization

** Setup
* PostgreSQL 9.6
  * extensions installed
CREATE EXTENSION postgis;
CREATE EXTENSION fuzzystrmatch; --needed for postgis_tiger_geocoder
CREATE EXTENSION address_standardizer;
CREATE EXTENSION address_standardizer_data_us;
CREATE EXTENSION postgis_topology;
CREATE EXTENSION postgis_tiger_geocoder;
  * Follow the instructions in database folder to import the data.

## directory structure
<pre>
├── README.md          <- The top-level README for developers using this project.
│
├── data               <- data need be downloaded elsewhere, only description here
│   ├── README.md      <- data source description
│
├── database           <- SQL script to import and process the data
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── notebooks
│   ├── EDA            <- descriptive analysis
│   ├── working        <- working notebooks for experiments, testing and debugging.
│   └── modeling       <- notebooks for modeling evolution and regression
│
├── viz                <- visualization dumps
│
├── src                <- python modules developped for both data processing, modeling and visualization
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│
└── misc

</pre>


