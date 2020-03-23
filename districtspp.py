# -*- coding: utf-8 -*-
"""
This Python module is designed to allow for easy simulation of gerrymandering algorithms.

Author: Salvador Balkus
"""

import shapely.geometry as shp
import numpy as np
import pandas as pd
import geopandas as gpd
import math as m
import requests
from scipy.spatial import Voronoi

#May want to change these methods in the future to take in
#FIPS instead of "state name"
fips_dict = {
 "Alabama" : "01",
 "Alaska" : "02",
 "Arizona" : "04",
 "Arkansas" : "05",
 "California" : "06",
 "Colorado" : "08",
 "Connecticut" : "09",
 "Delaware" : "10",
 "Florida" : "12",
 "Georgia" : "13",
 "Hawaii" : "15",
 "Idaho" : "16",
 "Illinois" : "17",
 "Indiana" : "18",
 "Iowa" : "19",
 "Kansas" : "20",
 "Kentucky" : "21",
 "Louisiana" : "22",
 "Maine" : "23",
 "Maryland" : "24",
 "Massachusetts" : "25",
 "Michigan" : "26",
 "Minnesota" : "27",
 "Mississippi" : "28",
 "Missouri" : "29",
 "Montana" : "30",
 "Nebraska" : "31",
 "Nevada" : "32",
 "New Hampshire" : "33",
 "New Jersey" : "34",
 "New Mexico" : "35",
 "New York" : "36",
 "North Carolina" : "37",
 "North Dakota" : "38",
 "Ohio" : "39",
 "Oklahoma" : "40",
 "Oregon" : "41",
 "Pennsylvania" : "42",
 "Rhode Island" : "44",
 "South Carolina" : "45",
 "South Dakota" : "46",
 "Tennessee" : "47",
 "Texas" : "48",
 "Utah" : "49",
 "Vermont" : "50",
 "Virginia" : "51",
 "Washington" : "53",
 "West Virginia" : "54",
 "Wisconsin" : "55",
 "Wyoming" : "56"

 }

districts_dict = {
 "Alabama" : 7,
 "Alaska" : 1,
 "Arizona" : 9,
 "Arkansas" : 4,
 "California" : 53,
 "Colorado" : 7,
 "Connecticut" : 5,
 "Delaware" : 1,
 "Florida" : 27,
 "Georgia" : 14,
 "Hawaii" : 2,
 "Idaho" : 2,
 "Illinois" : 18,
 "Indiana" : 9,
 "Iowa" : 4,
 "Kansas" : 4,
 "Kentucky" : 6,
 "Louisiana" : 6,
 "Maine" : 2,
 "Maryland" : 8,
 "Massachusetts" : 9,
 "Michigan" : 14,
 "Minnesota" : 8,
 "Mississippi" : 4,
 "Missouri" : 8,
 "Montana" : 1,
 "Nebraska" : 3,
 "Nevada" : 4,
 "New Hampshire" : 2,
 "New Jersey" : 12,
 "New Mexico" : 3,
 "New York" : 27,
 "North Carolina" : 13,
 "North Dakota" : 1,
 "Ohio" : 16,
 "Oklahoma" : 5,
 "Oregon" : 5,
 "Pennsylvania" : 18,
 "Rhode Island" : 2,
 "South Carolina" : 7,
 "South Dakota" : 1,
 "Tennessee" : 9,
 "Texas" : 36,
 "Utah" : 4,
 "Vermont" : 1,
 "Virginia" : 11,
 "Washington" : 10,
 "West Virginia" : 3,
 "Wisconsin" : 8,
 "Wyoming" : 1
        }

"""
Below are the original file-reading functions that were used to read in downloaded data.
These functions are old; instead, use the functions that access data from the Census API directly.
"""

def pop_read_file(path):
    """
    Produces clean DataFrame of population data from downloaded US Census Bureau file.
    """
    pop = pd.read_csv(path)
    pop = pop.drop(0)
    pop = pop.rename(columns = {"GEO_ID":"ID","NAME":"Name","B01003_001E":"Population","B01003_001M":"moe"})
    pop.Name = pop.Name.str.extract(r"([a-zA-Z -]*,)")[0].str[:-6]
    pop.Name = pop.Name.str.strip()
    pop.Population = pop.Population.astype(int)
    pop.loc[pop.Name.str.contains(" Town"), "Name"] = pop[pop.Name.str.contains(" Town")].Name.str[:-5]
    return pop

def pop_read_file_state(path, state_name):
    """
    Produces clean DataFrame of population data from downloaded US Census Bureau file of state population.
    """
    pop = pd.read_csv(path)
    pop = pop.drop(0)
    pop = pop.rename(columns = {"GEO_ID":"ID","NAME":"Name","B01003_001E":"Population","B01003_001M":"moe"})
    state_pop = pop[pop.Name.str.contains(state_name)]
    state_pop.Name = state_pop.Name.str[0:25]
    state_pop.Name = state_pop.Name.str.strip()
    state_pop.Population = state_pop.Population.astype(int)
    return state_pop

def geo_read_file(district_path, state_path, state_name):
    """
    Produces clean GeoDataFrame of specified district from downloaded TIGER Database .shp file.
    """
    geos = gpd.read_file(district_path)
    states = gpd.read_file(state_path)
    state_geo = geos[geos["STATEFP"] == fips_dict[state_name]]
    state_coast = states[states["NAME"] == state_name] 
    state = gpd.overlay(state_coast, state_geo, how="intersection")
    state.loc[state.NAME_2.str.contains(" Town"), "NAME_2"] = state[state.NAME_2.str.contains(" Town")].NAME_2.str[:-5]
    return state

def state_read_file(state_path, state_name):
    """
    Obtains boundary of specified state from downloaded TIGER Database .shp file.
    """
    states = gpd.read_file(state_path)
    state = states[states.NAME == state_name]
    state = state[["NAME","geometry"]].reset_index(drop = True)
    return state

"""
The functions below obtain data directly from the US Census API.
They are much easier to use, and are the functions used for the final analysis.
"""

def pop_read_api(state):
    """
    Obtains the population of each county subdivision of the specified state and performs data cleaning.
    
    INPUT: String of desired state name
    
    OUTPUT: DataFrame of population for each county subdivision
    """
    table = "S0101_C01_001E"
    api_call = "https://api.census.gov/data/2018/acs/acs5/subject?get=NAME," + table + "&for=county%20subdivision:*&in=county:*+state:" + fips_dict[state]
    pop = pd.DataFrame(requests.get(api_call).json())
    
    pop.columns = pop.loc[0]
    pop = pop.drop(0)
    
    pop = pop.rename(columns = {"NAME":"Name","S0101_C01_001E":"Population"})
    pop.Name = pop.Name.str.extract(r"([a-zA-Z -]*,)")[0].str[:-1]
    pop.Name = pop.Name.str.strip()
    pop.Population = pop.Population.astype(int)
    pop.loc[pop.Name.str.contains(" Town"), "Name"] = pop[pop.Name.str.contains(" Town")].Name.str[:-5]
    return pop
    

def state_read_api(state):
    """
    Obtains the shape file information of the specified state.
    
    INPUT: String of desired state name
    OUTPUT: GeoDataFrame of the state
    """
    
    api_call = "https://geo.dot.gov/server/rest/services/NTAD/States/MapServer/0/query?where=UPPER(STATEFP)%20like%20%27%25"+fips_dict[state]+"%25%27&outFields=*&outSR=4326&f=json"
    resp = requests.get(api_call).json()
    state = gpd.GeoDataFrame(resp["features"][0]["attributes"], index = [0], crs = {'init': 'epsg:4326'})
    state["geometry"] = [shp.MultiPolygon([shp.Polygon(shape) for shape in resp["features"][0]["geometry"]["rings"]])]
    
    return state

def geo_read_api(state):
    """
    Obtains the shape file information for each county subdivision of the specified state.
    
    INPUT: String of desired state name
    OUTPUT: GeoDataFrame of shape file for each county subdivision
    """
    api_call = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Places_CouSub_ConCity_SubMCD/MapServer/find?searchText="+fips_dict[state]+"&contains=true&searchFields=STATE&sr=&layers=1&returnGeometry=true&f=json"
    
    resp = requests.get(api_call).json()
    df = gpd.GeoDataFrame(columns = resp["results"][0]["attributes"].keys(), crs = {'init': 'epsg:3857'})
    
    geometries = []
    for n, result in enumerate(resp["results"]):
        df = df.append(pd.DataFrame(result["attributes"], index = [n]))
        geometries.append(shp.MultiPolygon([shp.Polygon(shape) for shape in result["geometry"]["rings"]]))
    
    df["geometry"] = gpd.GeoSeries(geometries)   
    df = df.to_crs({'init': 'epsg:4326'})

    state_bound = state_read_api(state)
    
    return gpd.overlay(df, state_bound, how="intersection")



"""
The functions below are used to produce spatial point processes of varying densities.
They are used to test the effects of point distribution on gerrymandering methods.
"""

def get_density(state_geo, state_pop, district_name):
    """
    Calculates the population density of a given geography.
    """
    district_geo = state_geo[state_geo["NAME_2"] == district_name] #used to be NAMELSAD
    density = state_pop[state_pop.Name == district_name].iloc[0,2] / district_geo.geometry.area.sum()
    return density

def uniform_density(geo, pop, district, density = 100000, file_name = "latest.shp", output_file = False):
    """
    Generates a GeoDataFrame of uniformly distributed points within the specified geography.
    Geography is specified via "district"
    """
    test = geo[geo["NAME_2"] == district] #in original, this was NAMELSAD
    xmin = test.bounds.minx.min()
    ymin = test.bounds.miny.min()
    xmax = test.bounds.maxx.max()
    ymax = test.bounds.maxy.max()

    width =  xmax - xmin
    height = ymax - ymin
    
    width_pop = m.sqrt(density * test.area.sum()) * (width / height)
    height_pop = m.sqrt(density * test.area.sum()) * (height / width)
    
    dist_width = width / width_pop
    dist_height = height / height_pop
    
    x = np.arange(xmin, xmax, dist_width)
    y = np.arange(ymin, ymax, dist_height)
    
    points = []
    grid = np.meshgrid(x,y)
    for n in range(len(grid[0])):
        points += zip(grid[0][n], grid[1][n])
        
    series = gpd.GeoSeries([shp.Point(x,y) for x,y in points])
    series = gpd.GeoDataFrame(series, crs = {'init':'epsg:4326'})
    series.columns = ["geometry"]
    
    intersection = series[series.intersects(test.geometry.iloc[0])]
    if(output_file):
        intersection.to_file(file_name)
    return intersection

"""
The functions below produce spatial point processes within the boundary of each state using files saved to the user's computer.
These functions are outdated; it is recommended to use the api functions, which are easier to call and do not required downloading.
"""

def district_spp(pop_path, geo_path, state_path, state, district, density_frac = 0.01, mask_plot = False):
    pop = pop_read_file(pop_path) #used to use the original pop_data_read
    geo = geo_read_file(geo_path, state_path, state)
    density = get_density(geo, pop,district)
    region = uniform_density(geo, pop, district, density_frac * density )
    
    if(not mask_plot):
        region.plot(markersize = 0.5)
    return region

def state_spp(pop_path, geo_path, state_path, state, density_frac = 0.001, mask_plot = False, file_name = "latest_state.shp"):
    pop = pop_read_file(pop_path, state)
    geo = geo_read_file(geo_path, state_path, state)
    
    district = "Congressional District 1"

    points = [uniform_density(geo,pop,district,density_frac * get_density(geo,pop,district))]

    for n in range(2,districts_dict[state]+1):
        district = "Congressional District " + str(n)
        points.append(uniform_density(geo,pop,district,density_frac * get_density(geo,pop,district)))
    
    full_state = gpd.GeoDataFrame(pd.concat(points, ignore_index=True), crs=points[0].crs)
    if(not mask_plot):
        full_state.plot(markersize = 1)
        
    full_state.to_file(file_name)
    return full_state

def state_spp2(pop_path, geo_path, state_path, state, density_frac = 0.001, mask_plot = False, file_name = "latest_state.shp", sz = 1):
    pop = pop_read_file(pop_path)
    geo = geo_read_file(geo_path, state_path, state)
    locations = pop.Name
    points = []

    for location in locations:
        try:
            if(location != "County subdivisions not de"):
                points.append(uniform_density(geo,pop,location,density_frac * get_density(geo,pop,location)))
        except:
            print(location)
    
    full_state = gpd.GeoDataFrame(pd.concat(points, ignore_index=True), crs=points[0].crs)
    if(not mask_plot):
        full_state.plot(markersize = sz)
        
    full_state.to_file(file_name)
    return full_state

"""
These functions use the Census API to produce spatial point processes of the population of the specified state.
"""

def state_spp_api(state):
    pop = pop_read_api(state)
    geo = geo_read_api(state)
    pass





def knn_district(spp, state_boundary, state):
    """
    Takes in a spatial point process and allocates each point into a district, labeling the point with a number representing the district into which it has been placed..
    Districts are created by first selecting the furthest population point from the centroid of the state, then allocating the closest n population points to the selected point,
    where n represents the number of population points set to be allocated to each state congressional district. 
    
    INPUT
        spp: a GeoDataFrame of points representing a spatial point process of a population.
        state_boundary: a GeoDataFrame representing the boundary of the state.
        state: the name of the state, represented as a string.
    OUTPUT
        a GeoDataFrame of points labeled with a number representing the district which the point was placed in
    """
    district_count = districts_dict[state]
    target = round(len(spp) / district_count)
    clusters = []
    spp["centroid_dist"] = spp.distance(state_boundary.centroid[0])
    test = spp.loc[spp.centroid_dist.idxmax()].geometry
    
    remaining = spp.copy()
    
    for n in range(district_count):
        remaining["cluster_dist"] = remaining.distance(test)
        if(len(remaining) > target):
            sorted_remaining = remaining.sort_values(by = "cluster_dist")
            clusters.append(sorted_remaining.iloc[0:target].index)
            remaining = remaining.loc[sorted_remaining.iloc[target:].index]
            test = remaining.loc[remaining.centroid_dist.idxmax()].geometry
        else:
            clusters.append(remaining.index)
            
    spp["district"] = [-1]*len(spp)
    
    for n, cluster in enumerate(clusters):
        spp.loc[cluster, "district"] = n
    
    return spp

def voronoi_district(spp, state_boundary, state):
    """
    Takes in a labeled spatial point process of population points and outputs the shape of each district. 
    
    INPUT
        spp: GeoDataFrame of population points. Must contain a column "geometry" and a column "district" with districts labeled by number.
        state_boundary: GeoDataFrame with a polygon of the state boundary
        state: the name of the state, represented as a string.
    OUTPUT
        GeoDataFrame of polygons, each representing a congressional district.
    
    """
    district_count = districts_dict[state]
    test_points = list(zip(spp.geometry.x, spp.geometry.y))
    v = Voronoi(test_points)
    polygons = {}
    for id, region_index in enumerate(v.point_region):
        points = []
        for vertex_index in v.regions[region_index]:
            if vertex_index != -1:  # the library uses this for infinity
                points.append(list(v.vertices[vertex_index]))
        points.append(points[0])
        polygons[id]=points
    
    gons = []
    for n in range(len(polygons)):
        gons.append(shp.Polygon(polygons[n]))
    
    vs = gpd.GeoDataFrame(geometry = gons)
    vs["district"] = spp.district
    vs_valid = vs[vs["geometry"].area != 0]
    inter = gpd.overlay(vs_valid, state_boundary, how = "intersection")
    
    districts = []
    for n in range(district_count):
        d = inter[inter["district"] == n]
        districts.append(d.geometry.unary_union)
    
    final = gpd.GeoDataFrame(geometry = districts).reset_index()
    final.columns = ["district","geometry"]
    return final
        
    
    
    
    