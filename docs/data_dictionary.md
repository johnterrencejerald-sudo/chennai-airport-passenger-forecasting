# Data dictionary

The analysis uses monthly domestic aviation-traffic information for Chennai International Airport from the OpenCity urban-data portal.

| Variable | Description | Analytical use |
|---|---|---|
| `Year` | Calendar year associated with the traffic record | Combined with month to construct the time index |
| `Month` | Calendar month associated with the traffic record | Combined with year to construct the time index |
| `Date` | Derived monthly date index | Time-series index after preprocessing |
| `Pax To Origin` | Monthly domestic passenger-flow measure for Chennai Airport, as named in the source data | Primary outcome/forecast target |
| `Pax From Origin` | Monthly domestic passenger-flow measure for Chennai Airport, as named in the source data | Exogenous predictor in SARIMAX |
| Freight variables | Freight traffic measures to and from origin | Excluded because of substantial missingness |
| Mail variables | Mail traffic measures to and from origin | Excluded because of substantial missingness |

## Data source

OpenCity urban-data portal: [Chennai aviation traffic data](https://data.opencity.in/dataset/chennai-aviation-traffic-data).

## Data-use note

The repository will provide data-access and preprocessing instructions alongside reproducible code. Any inclusion of source data will follow the source portal's licensing and redistribution terms.
