# Mobile Cellular traces - Irish case

This list of files aims to provide an example on how to combine demographic data and mobile networks deployment
to help researchers to study the planning of mobile cellular networks.
The data have been anonymized.

# Files:
    - points_$MAX_POPULATION_$MAX_AREA.csv
    - base_stations.csv
    - adj_$MAX_POPULATION_$MAX_AREA.csv
    - cdf_mno_1_$AREA_TYPE.csv
    - cdf_mno_2_$AREA_TYPE.csv

# Description of the files:
    - points_$MAX_POPULATION_$MAX_AREA.csv:
        List of subscriber clusters. Each represents at most $MAX_POPULATION population and $MAX_AREA area.
        They are created from the polygons provided by the Irish Central Statistics Office.
        If the population of a polygon < $MAX_POPULATION and the area of a polygon < $MAX_AREA then the
        polygon is represented with its centroid.
        Otherwise, the polygon is represented with as many points (subscriber clusters) necessary to divide equally
        the polygon population and polygon area to satisfy the above conditions.
        The area types {urban, suburban, rural} are defined on a county level depending on the density of
        the population per km^2. All the points (subscriber clusters) belonging to a particular county will
        have the same area type. In this case study:
            if a county has a population density per km^2 < 100 is defined as rural (coded as ‘rural’)
            if a county has a population density per km^2 >=100 and < 1000 is defined as suburban (coded as ‘suburban’)
            if a county has a population density per km^2 >=1000 is defined as urban (coded as ‘urban’)
        
    - base_stations.csv:
        List of base stations. They can be found on the Irish telecommunications regulator website.
    
    - adj_$MAX_POPULATION_$MAX_AREA.csv
        List of adjacencies in the downlink case representing the pairs (base station, subscriber cluster) within the coverage
        range. The sensitivity device has been set to -105 dBm.
    
    - cdf_mno_X_$AREA_TYPE:
        $AREA_TYPE corresponds to {urban, suburban, rural}. They correspond to the source code of the CDFs traffic demand for mno X (either 1 or 2).

# List of parameters:
    - points_$MAX_POPULATION_$MAX_AREA.csv
        - p_id:             subscriber cluster ID
        - p_lat:            latitude of p_id
        - p_long:           longitude of p_id
        - p_area:           p_id area (in km^2)
        - p_population:     p_id population
        - p_county:         county associated with p_id
        - p_type:           p_id area type - {‘urban’, ‘suburban’, ‘rural’}
    - base_stations.csv:
        - bs_id:            base station ID
        - bs_technology:    bs_id technology {gsm,3g}
        - bs_fc:            bs_id transmitting frequency
        - bs_operator:      bs_id operator {mno1,mno2}
        - bs_lat:           bs_id latitude
        - bs_long:          bs_id longitude
        - bs_power:         bs_id transmitting power (in dBm)
    - adj_$MAX_POPULATION_$MAX_AREA.csv
        - bs_id:            base station ID
        - p_id:             subscriber cluster ID
        - bp_distance:      distance between bs_id and p_id (in km)
        - bp_attenuation:   attenuation between bs_id and p_id (in dB)
        - bp_rssi:          received signal strength between bs_id and p_id (in dBm)
	- bp_sinr:	    sinr between bs_id and p_id (in dB) (N.B. it depends on frequency (only base stations operating on the same frequency interfere), frequency reuse (reuse factor 1 in this case), and operator (only base stations belonging to the same operators interfere))

