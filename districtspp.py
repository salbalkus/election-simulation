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
import os

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



class StateSPP:

    """
    The functions below obtain data directly from the US Census API.
    They are much easier to use, and are the functions used for the final analysis.
    """

    def __init__(self, state, api_key = "", file_name = "spp.shp"):
        self.state = state
        self.api_key = api_key
        wd = os.getcwd()
        if self.state not in os.listdir(wd):
            os.mkdir(self.state)
        self.wd = wd + "//" + state
        
        if file_name in os.listdir(self.wd):
            self.spp = gpd.read_file(self.wd + "//" + file_name)
            self.spp = self.spp.drop("FID", axis = 1)
        else:
            self.spp = None
        
        self.boundary = self.state_read_api()

        if api_key == "":
            self.pop = self.pop_read_api()
            self.geo = self.geo_read_api()
        else:
            self.pop = self.pop_read_api()
            self.geo = self.geo_read_api()
        
        self.districts = self.districts_read_api()
        
        
    def save_progress(self, file_name = "spp.shp"):
        if type(self.spp) == "NoneType":
            print("Nothing to save.")
        else:
            self.spp.to_file(self.wd + "//" + file_name)
        


    def pop_read_api(self):
        """
        Obtains the population of each county subdivision of the specified state and performs data cleaning.
        
        INPUT: String of desired state name
        
        OUTPUT: DataFrame of population for each county subdivision
        """
        table = "S0101_C01_001E"
        api_call = "https://api.census.gov/data/2018/acs/acs5/subject?get=NAME," + table + "&for=county%20subdivision:*&in=county:*+state:" + fips_dict[self.state] + "&key=" + self.api_key
        pop = pd.DataFrame(requests.get(api_call).json())
        
        pop.columns = pop.loc[0]
        pop = pop.drop(0)
        
        pop = pop.rename(columns = {"NAME":"Name","S0101_C01_001E":"Population"})
        pop.Name = pop.Name.str.extract(r"([a-zA-Z .'-]*,)")[0].str[:-1]
        pop.Name = pop.Name.str.strip()
        pop.Population = pop.Population.astype(int)
        pop = pop[pop.Name != "County subdivisions not defined"]
        return pop
    
    def districts_read_api(self):
        """
        Obtains the the shape file for the congressional districts of the specified state
        """
        api_call = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_ACS2019/MapServer/find?searchText="+fips_dict[self.state]+"&contains=true&searchFields=STATE&sr=&layers=54&returnGeometry=true&f=json&key=" + self.api_key
        resp = requests.get(api_call).json()
        df = gpd.GeoDataFrame(columns = resp["results"][0]["attributes"].keys(), crs = {'init': 'epsg:3857'})
        geometries = []
        for n, result in enumerate(resp["results"]):
            df = df.append(pd.DataFrame(result["attributes"], index = [n]))
            geometries.append(shp.MultiPolygon([shp.Polygon(shape) for shape in result["geometry"]["rings"]]))
        
        df["geometry"] = gpd.GeoSeries(geometries)   
        df = df.to_crs({'init': 'epsg:4326'})
    
        state_bound = self.boundary
        
        return gpd.overlay(df, state_bound, how="intersection")
    
    def state_read_api(self):
        """
        Obtains the shape file information of the specified state.
        
        INPUT: String of desired state name
        OUTPUT: GeoDataFrame of the state
        """
        
        api_call = "https://geo.dot.gov/server/rest/services/NTAD/States/MapServer/0/query?where=UPPER(STATEFP)%20like%20%27%25"+fips_dict[self.state]+"%25%27&outFields=*&outSR=4326&f=json"
        resp = requests.get(api_call).json()
        boundary = gpd.GeoDataFrame(resp["features"][0]["attributes"], index = [0], crs = {'init': 'epsg:4326'})
        boundary["geometry"] = [shp.MultiPolygon([shp.Polygon(shape) for shape in resp["features"][0]["geometry"]["rings"]])]
        return boundary
    
    def geo_read_api(self):
        """
        Obtains the shape file information for each county subdivision of the specified state.
        
        INPUT: String of desired state name
        OUTPUT: GeoDataFrame of shape file for each county subdivision
        """
        api_call = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Places_CouSub_ConCity_SubMCD/MapServer/find?searchText="+fips_dict[self.state]+"&contains=true&searchFields=STATE&sr=&layers=1&returnGeometry=true&f=json&key=" + self.api_key
        
        resp = requests.get(api_call).json()
        df = gpd.GeoDataFrame(columns = resp["results"][0]["attributes"].keys(), crs = {'init': 'epsg:3857'})
        
        geometries = []
        for n, result in enumerate(resp["results"]):
            df = df.append(pd.DataFrame(result["attributes"], index = [n]))
            geometries.append(shp.MultiPolygon([shp.Polygon(shape) for shape in result["geometry"]["rings"]]))
        
        df["geometry"] = gpd.GeoSeries(geometries)   
        df = df.to_crs({'init': 'epsg:4326'})        
    
        df = df[df.NAME != "County subdivisions not defined"]
        return gpd.overlay(df, self.boundary, how="intersection")
        
    
    """
    The functions below are used to produce spatial point processes of varying densities.
    They are used to test the effects of point distribution on gerrymandering methods.
    """
    
    """
    The following two functions generate a uniform array of points
    """
    def get_density(self,cousub):
        """
        Calculates the population density of a given geography.
        """
        district_geo = self.geo[self.geo.COUSUB == cousub]
        density = self.pop[self.pop["county subdivision"] == cousub].iloc[0,1] / district_geo.geometry.area.sum()
        return density
    
    def uniform_density(self,cousub, density = 100000):
        """
        Generates a GeoDataFrame of uniformly distributed points within the specified geography.
        Geography is specified via "cousub"
        """
        print(cousub + " in progress")
        test = self.geo[self.geo.COUSUB == cousub]
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

    def normalize(self,x):
        return (x - x.min()) / (x.max()-x.min())
    
    def weight_cousub(self,cousub):
        """
        Weights each town based on population and distance from the selected county subdivision.
        Used in 'distance_based_density' to select a town to generate the spatial point process.
        """
        
        select = gpd.GeoDataFrame(self.geo[["COUSUB","geometry"]].merge(self.pop[["county subdivision","Population"]], how = "inner", left_on = "COUSUB",right_on = "county subdivision").drop("county subdivision", axis = 1))
        select["centroid_distance"] = select.centroid.distance(select[select["COUSUB"] == cousub].centroid.iloc[0])
        select = select[select["COUSUB"] != cousub]
        select["Population"] = self.normalize(select.Population.values)
        
        ratio = select.Population / select.centroid_distance
        select["p1"] = ratio / ratio.sum()
        
        cumulative = 0
        cp1 = []
        
        for p in select.p1.values:
            cumulative += p
            cp1.append(cumulative)
            
        select["cp1"] = pd.Series(cp1)
        
        return select
    
    def distance_pdf(self,point, centroid, centroid_dist):
        """
        Calculates a value based on how far away a given point is from the centroid of a town.
        The farther away, the less likely the point is to be generated.
        Note that this is not a valid pdf, as values are not limited to the [0,1] interval.
        It can only be sampled from using the Metropolis-Hastings algorithm
        """
        return centroid_dist/(((point.x-centroid.x)**2 + (point.y-centroid.y)**2)**4)
    
    
    
    def pick_point_start(self,geo, xmin, xmax, ymin, ymax):
        """
        Uses rejection sampling to randomly select a point within the given region
        """
        while(True):
            point = shp.Point(uniform(xmin, xmax), uniform(ymin, ymax))
            if geo.geometry.values[0].contains(point):
                return point
    
    #This is broken, must adjust to work better
    def pick_point(self, geo, prev, fails, dist):
        """
        Uses rejection sampling to randomly select a point within the given region
        """
        while(True):
            r = (((random()*2)-1)*dist)/fails
            theta = random()*2*m.pi
            point = shp.Point(prev.x + m.cos(theta)*r, prev.y + m.sin(theta)*r)
            if geo.geometry.values[0].contains(point):
                return point
    
    
    def distance_based_density(self, cousub, density_frac = 0.01):
        """
        Generates a GeoDataFrame of uniformly distributed points within the specified geography.
        Geography is specified via "cousub"
        """
        print(cousub + " in progress")

        test_geo = self.geo[self.geo.COUSUB == cousub] 
        test_pop = self.pop[self.pop["county subdivision"] == cousub]
        xmin = test_geo.bounds.minx.min()
        ymin = test_geo.bounds.miny.min()
        xmax = test_geo.bounds.maxx.max()
        ymax = test_geo.bounds.maxy.max()
            
        points_to_place = test_pop.Population.values[0] * density_frac
        point1 = self.pick_point_start(test_geo,xmin,xmax,ymin,ymax)
        point2 = None
        
        points = []
        towns = self.weight_cousub(cousub)
        town = towns.loc[towns.p1.idxmax()]
        centroid_dist = test_geo.geometry.centroid.distance(town.geometry.centroid)
        fails = 1
    
        while(len(points) < points_to_place):
            
            if fails == 1:
                point2 = self.pick_point_start(test_geo,xmin,xmax,ymin,ymax)
            else:
                point2 = self.pick_point(test_geo, point1, fails, max([xmax-xmin,ymax-ymin]))
                    
            point1_p = float(self.distance_pdf(point1, town.geometry.centroid, centroid_dist))
            point2_p = float(self.distance_pdf(point2, town.geometry.centroid, centroid_dist))
            
            a = point2_p / point1_p            
            if a > random():
                points.append(point1)
                point1 = point2
                fails = 1
            else:
                fails += 1
        
        return gpd.GeoDataFrame(points, columns = ["geometry"])
    
    
    """
    These functions use the Census API to produce spatial point processes of the population of the specified state.
    """
    
    def generate_spp(self, spp_type = "distance", density_frac = 0.01, mask_plot = False, sz = 1):
        locations = self.pop["county subdivision"]
        points = []
        
        if spp_type == "distance":
            for location in locations:
                try:
                    points.append(self.distance_based_density(location, density_frac))
                except Exception as e:
                    print(location)
        else:
            for location in locations:
                try:
                    points.append(self.uniform_density(location, density = self.get_density(location) * density_frac))
                except:
                    print(location)
                    
        print(self.state + " complete")

        self.spp = gpd.GeoDataFrame(pd.concat(points, ignore_index = True), crs = points[0].crs)        
        if(not mask_plot):
            self.spp.plot(markersize = sz)
    
    
    def knn_district(self):
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
        
        print("Redistricting " + self.state)
        district_count = districts_dict[self.state]
        target = round(len(self.spp) / district_count)
        clusters = []
        self.spp["centroid_dist"] = self.spp.distance(self.boundary.centroid[0])
        test = self.spp.loc[self.spp.centroid_dist.idxmax()].geometry
        
        remaining = self.spp.copy() ## THIS COULD CAUSE POTENTIAL BUGS!!!
        
        for n in range(district_count):
            remaining["cluster_dist"] = remaining.distance(test)
            if(len(remaining) > target):
                sorted_remaining = remaining.sort_values(by = "cluster_dist")
                clusters.append(sorted_remaining.iloc[0:target].index)
                remaining = remaining.loc[sorted_remaining.iloc[target:].index]
                test = remaining.loc[remaining.centroid_dist.idxmax()].geometry
            else:
                clusters.append(remaining.index)
                
        self.spp["district"] = [-1]*len(self.spp)
        
        for n, cluster in enumerate(clusters):
            self.spp.loc[cluster, "district"] = n + 1
        
        
            
    def voronoi_district(self):
        """
        Takes in a labeled spatial point process of population points and outputs the shape of each district. 
        
        INPUT
            spp: GeoDataFrame of population points. Must contain a column "geometry" and a column "district" with districts labeled by number.
            state_boundary: GeoDataFrame with a polygon of the state boundary
            state: the name of the state, represented as a string.
        OUTPUT
            GeoDataFrame of polygons, each representing a congressional district.
        
        """
        district_count = districts_dict[self.state]
        test_points = list(zip(self.spp.geometry.x, self.spp.geometry.y))
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
        vs["district"] = self.spp.district
        vs_valid = vs[vs["geometry"].area != 0]
        inter = gpd.overlay(vs_valid, self.boundary, how = "intersection")
        
        districts = []
        for n in range(district_count):
            d = inter[inter["district"] == n]
            districts.append(d.geometry.unary_union)
        
        final = gpd.GeoDataFrame(geometry = districts).reset_index()
        final.columns = ["district","geometry"]
        final.to_file(self.wd + "//districts.shp")
        self.districts = final
        
    def read_votes(self,file_path):
        votes = pd.read_csv(file_path)
        self.votes = votes[votes.state == self.state]
            
    def assign_district_p(self, file_path):
        """
        Assigns probabilities of party votes to each point in a spatial point process.
        Relies on a file from MIT Election Data providing the number of votes for each party in each congressional district in the 2018 election.
        """
        output = self.spp.copy()
        output["dem_p"] = 0.0
        output["rep_p"] = 0.0
        output["lib_p"] = 0.0
        output["grn_p"] = 0.0
        
        districts_list = list(self.districts.BASENAME)
        
        for n in districts_list:
            points = output.within(self.districts.loc[int(n)-1,"geometry"])
            district_votes = self.votes[self.votes.district == n][["votes","party"]]
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
        
        self.spp = output
    
    def cast_vote(self):
        """
        Casts a vote for a single population point randomly based on proportions of voters in the point's congressional district.
        Helper function for simulate_voting
        """
        x = random()
        thresh = self.spp.dem_p
        if(x < thresh):
            return "democrat"
        else:
            thresh += self.spp.rep_p
            if(x < thresh):
                return "republican"
            else:
                thresh += self.spp.lib_p
                if(x < thresh):
                    return "libertarian"
                else:
                    thresh += self.spp.grn_p
                    if(x < thresh):
                        return "green"
                    else:
                        return "other"
            
    def simulate_voting(self):
        """
        Simulates a possible election for a spatial point process based on voting proportions across the state for each congressional district.
        """
        output = self.spp.copy()
        output["vote"] = output.apply(self.cast_vote, axis = 1,result_type = "expand")
        return output
    
    def run_sims(self, file_name = "simulation.csv", sims = 10):
        """
        Runs many simulations and summarizes the number of votes for each in a Data Frame
        """
        safe_spp = self.spp.copy()
        sim = self.simulate_voting(safe_spp)
        output = pd.DataFrame(sim.groupby(["district","vote"]).size()).T
        for n in range(1,sims):
            sim = self.simulate_voting(safe_spp)
            output = output.append(pd.DataFrame(sim.groupby(["district","vote"]).size()).T)
        
        output = output.reset_index()
        output.to_csv(file_name)
        self.simulations = output
    


    
        
    
    
    