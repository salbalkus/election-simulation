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
from random import random, uniform


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
The functions below obtain data directly from the US Census API.
They are much easier to use, and are the functions used for the final analysis.
"""

def pop_read_api(state, api_key = ""):
    """
    Obtains the population of each county subdivision of the specified state and performs data cleaning.
    
    INPUT: String of desired state name
    
    OUTPUT: DataFrame of population for each county subdivision
    """
    table = "S0101_C01_001E"
    api_call = "https://api.census.gov/data/2018/acs/acs5/subject?get=NAME," + table + "&for=county%20subdivision:*&in=county:*+state:" + fips_dict[state] + "&key=" + api_key
    pop = pd.DataFrame(requests.get(api_call).json())
    
    pop.columns = pop.loc[0]
    pop = pop.drop(0)
    
    pop = pop.rename(columns = {"NAME":"Name","S0101_C01_001E":"Population"})
    pop.Name = pop.Name.str.extract(r"([a-zA-Z .-]*,)")[0].str[:-1]
    pop.Name = pop.Name.str.strip()
    pop.Population = pop.Population.astype(int)
    pop = pop[pop.Name != "County subdivisions not defined"]
    return pop

def districts_read_api(state, api_key = ""):
    """
    Obtains the the shape file for the congressional districts of the specified state
    """
    api_call = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_ACS2019/MapServer/find?searchText="+fips_dict[state]+"&contains=true&searchFields=STATE&sr=&layers=54&returnGeometry=true&f=json&key=" + api_key
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

def geo_read_api(state, api_key = ""):
    """
    Obtains the shape file information for each county subdivision of the specified state.
    
    INPUT: String of desired state name
    OUTPUT: GeoDataFrame of shape file for each county subdivision
    """
    api_call = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Places_CouSub_ConCity_SubMCD/MapServer/find?searchText="+fips_dict[state]+"&contains=true&searchFields=STATE&sr=&layers=1&returnGeometry=true&f=json&key=" + api_key
    
    resp = requests.get(api_call).json()
    df = gpd.GeoDataFrame(columns = resp["results"][0]["attributes"].keys(), crs = {'init': 'epsg:3857'})
    
    geometries = []
    for n, result in enumerate(resp["results"]):
        df = df.append(pd.DataFrame(result["attributes"], index = [n]))
        geometries.append(shp.MultiPolygon([shp.Polygon(shape) for shape in result["geometry"]["rings"]]))
    
    df["geometry"] = gpd.GeoSeries(geometries)   
    df = df.to_crs({'init': 'epsg:4326'})
    state_bound = state_read_api(state)
    

    df = df[df.NAME != "County subdivisions not defined"]
    return gpd.overlay(df, state_bound, how="intersection")
    



"""
The functions below are used to produce spatial point processes of varying densities.
They are used to test the effects of point distribution on gerrymandering methods.
"""

"""
The following two functions generate a uniform array of points
"""
def get_density(state_geo, state_pop, cousub):
    """
    Calculates the population density of a given geography.
    """
    district_geo = state_geo[state_geo.COUSUB == cousub]
    density = state_pop[state_pop["county subdivision"] == cousub].iloc[0,1] / district_geo.geometry.area.sum()
    return density

def uniform_density(cousub, geo, pop, density = 100000):
    """
    Generates a GeoDataFrame of uniformly distributed points within the specified geography.
    Geography is specified via "cousub"
    """
    test = geo[geo.NAME_1 == cousub]
    xmin = test.bounds.minx.min()
    ymin = test.bounds.miny.min()
    xmax = test.bounds.maxx.max()
    ymax = test.bounds.maxy.max()

    width =  xmax - xmin
    height = ymax - ymin
    total = width * height
    
    width_pop = m.sqrt(density * total) * (width / height)
    height_pop = m.sqrt(density * total) * (height / width)
    
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
    return intersection

"""
The following five functions generate points such that the spatial point process
is weighted towards large cities
"""

def normalize(x):
    """
    Helper function for 'weight_towns'
    """
    return (x - x.min()) / (x.max()-x.min())

def weight_cousub(cousub, geo, pop):
    """
    Weights each town based on population and distance from the selected county subdivision.
    Used in 'distance_based_density' to select a town to generate the spatial point process.
    """
    
    select = gpd.GeoDataFrame(geo[["NAME_1","geometry"]].merge(pop[["Name","Population"]], how = "inner", left_on = "NAME_1",right_on = "Name").drop("NAME_1", axis = 1))
    select["centroid_distance"] = select.centroid.distance(select[select.Name == cousub].centroid.iloc[0])
    select = select[select["Name"] != cousub]
    select["Population"] = normalize(select.Population.values)
    
    ratio = select.Population / select.centroid_distance
    select["p1"] = ratio / ratio.sum()
    
    cumulative = 0
    cp1 = []
    
    for p in select.p1.values:
        cumulative += p
        cp1.append(cumulative)
        
    select["cp1"] = pd.Series(cp1)
    
    return select

def distance_pdf(point, centroid, centroid_dist):
    """
    Calculates a value based on how far away a given point is from the centroid of a town.
    The farther away, the less likely the point is to be generated.
    Note that this is not a valid pdf, as values are not limited to the [0,1] interval.
    It can only be sampled from using the Metropolis-Hastings algorithm
    """
    return centroid_dist/(((point.x-centroid.x)**2 + (point.y-centroid.y)**2)**4)


def pick_point(geo, xmin, xmax, ymin, ymax):
    """
    Uses rejection sampling to randomly select a point within the given region
    """
    while(True):
        point = shp.Point(uniform(xmin, xmax), uniform(ymin, ymax))
        if geo.geometry.values[0].contains(point):
            return point


def distance_based_density(cousub, geo, pop, density_frac = 0.001):
    """
    Generates a GeoDataFrame of uniformly distributed points within the specified geography.
    Geography is specified via "cousub"
    """
    test_geo = geo[geo.NAME_1 == cousub] 
    test_pop = pop[pop.Name == cousub]
    xmin = test_geo.bounds.minx.min()
    ymin = test_geo.bounds.miny.min()
    xmax = test_geo.bounds.maxx.max()
    ymax = test_geo.bounds.maxy.max()
        
    points_to_place = test_pop.Population.values[0] * density_frac
    point1 = pick_point(test_geo,xmin,xmax,ymin,ymax)
    point2 = None
    
    points = []
    towns = weight_cousub(cousub, geo, pop)
    town = towns.loc[towns.p1.idxmax()]

    while(len(points) < points_to_place):
        centroid_dist = test_geo.geometry.centroid.distance(town.geometry.centroid)
        
        point2 = pick_point(test_geo,xmin,xmax,ymin,ymax)  
                
        point1_p = float(distance_pdf(point1, town.geometry.centroid, centroid_dist))
        point2_p = float(distance_pdf(point2, town.geometry.centroid, centroid_dist))
        
        a = point2_p / point1_p
        
        if a > random():
            points.append(point1)
            point1 = point2        
    
    return gpd.GeoDataFrame(points)


"""
These functions use the Census API to produce spatial point processes of the population of the specified state.
"""

def state_spp_api(state, spp_type = "distance", density_frac = 0.001, mask_plot = False, file_name = "latest_state.shp", sz = 1, api_key = ""):
    if api_key == "":
        pop = pop_read_api(state)
        geo = geo_read_api(state)
    else:
        pop = pop_read_api(state, api_key)
        geo = geo_read_api(state, api_key)
    locations = pop.Name
    points = []
    
    if spp_type == "distance":
        for location in locations:
            try:
                points.append(distance_based_density(location, geo, pop, density_frac))
            except:
                print(location)
    else:
        for location in locations:
            try:
                points.append(uniform_density(location, geo, pop, density = get_density(geo,pop,location) * density_frac))
            except:
                print(location)
                
    
    full_state = gpd.GeoDataFrame(pd.concat(points, ignore_index = True), crs = points[0].crs)
    if(not mask_plot):
        full_state.plot(markersize = sz)
        
    full_state.to_file(file_name)
    return full_state
    


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
        spp.loc[cluster, "district"] = n + 1
    
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

def assign_district_p(spp, state, file_path, api_key = ""):
    """
    Assigns probabilities of party votes to each point in a spatial point process.
    Relies on a file from MIT Election Data providing the number of votes for each party in each congressional district in the 2018 election.
    """
    output = spp.copy()
    output["dem_p"] = 0.0
    output["rep_p"] = 0.0
    output["lib_p"] = 0.0
    output["grn_p"] = 0.0
    
    votes = pd.read_csv(file_path)
    votes = votes[votes.state == state]
    
    districts = districts_read_api(state, api_key)
    districts_list = list(districts.BASENAME)
    
    for n in districts_list:
        points = output.within(districts.loc[int(n)-1,"geometry"])
        district_votes = votes[votes.district == n][["votes","party"]]
        total_votes = sum(district_votes.votes)
        parties = list(district_votes.party)
        if "democratic" in parties:
            output.loc[points, "dem_p"] = float(district_votes[district_votes.party == "democratic"].iloc[0,0] / total_votes)
        if "republican" in parties:
            output.loc[points, "rep_p"] = float(district_votes[district_votes.party == "republican"].iloc[0,0] / total_votes)
        if "libertarian" in parties:
            output.loc[points, "lib_p"] = float(district_votes[district_votes.party == "libertarian"].iloc[0,0] / total_votes)
        if "green" in parties:
            output.loc[points, "grn_p"] = float(district_votes[district_votes.party == "green"].iloc[0,0] / total_votes)
    
    return output

def create_district_spp(state, file_path, density = 0.001, redistrict = knn_district, api_key = "", file_name = "latest.shp"):
    boundary = state_read_api(state)
    spp = state_spp_api(state, density_frac = density, api_key = api_key)
    spp = redistrict(spp, boundary, state)
    spp = assign_district_p(spp, state, file_path, api_key = api_key)
    spp.to_file(file_name)
    return spp

def cast_vote(spp):
    """
    Casts a vote for a single population point randomly based on proportions of voters in the point's congressional district.
    Helper function for simulate_voting
    """
    x = random()
    thresh = spp.dem_p
    if(x < thresh):
        return "democrat"
    else:
        thresh += spp.rep_p
        if(x < thresh):
            return "republican"
        else:
            thresh += spp.lib_p
            if(x < thresh):
                return "libertarian"
            else:
                thresh += spp.grn_p
                if(x < thresh):
                    return "green"
                else:
                    return "other"
        
def simulate_voting(spp):
    """
    Simulates a possible election for a spatial point process based on voting proportions across the state for each congressional district.
    """
    output = spp.copy()
    output["vote"] = output.apply(cast_vote, axis = 1,result_type = "expand")
    return output

def run_simulations(spp, sims = 10):
    """
    Runs many simulations and summarizes the number of votes for each in a Data Frame
    """
    safe_spp = spp.copy()
    sim = simulate_voting(safe_spp)
    output = pd.DataFrame(sim.groupby(["district","vote"]).size()).T
    for n in range(1,sims):
        sim = simulate_voting(safe_spp)
        output = output.append(pd.DataFrame(sim.groupby(["district","vote"]).size()).T)
    
    output = output.reset_index()
    return output



    
        
    
    
    
    