#! /usr/bin/env python

# author: pdifranc
# date: 06/09/2016
# description: mobile network planning for IEEE Transaction of Big Data: user clusters file

"""
0:OBJECTID
1:NUTS1
2:NUTS1NAME
3:NUTS2
4:NUTS2NAME
5:NUTS3
6:NUTS3NAME
7:COUNTY
8:COUNTYNAME
9:CSOED
10:OSIED
11:EDNAME
12:SMALL_AREA
13:GEOGID
14:MALE2011
15:FEMALE2011
16:TOTAL2011
17:PPOCC2011
18:UNOCC2011
19:VACANT2011
20:HS2011
21:PCVAC2011
22:CREATEDATE
23:Shape_Leng
24:Shape_Area
25:trasp
"""

dbf_file='./Ireland/SmallAreaConverted_GPS.dbf'
shp_file='./Ireland/SmallAreaConverted_GPS.shp'

path_traces = './traces'

import pandas as pd,pysal as ps
import shapely.geometry as sg
import ast
import random,argparse as ap
import numpy as np
import sys
import pyproj
import shapely.ops as ops
from functools import partial

# We need to change the default encoding to "latin-1"
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('latin-1')

# Class to generate the user clusters from the Irish census shapefiles
#
class Points():
  def __init__(self, max_pop, max_area):
    self.point_tuples = []
    self.maxsize = -1
    self.max_pop = max_pop
    self.max_area = max_area

  # computes the area of of a polygon which vertices are passed in GPS coordinates
  # @params[shapely.geometry.Polygon] polygon
  #
  # @return[Float] area in km^2
  #
  def area_from_coordinates(self, polygon):
    polygon_area = ops.transform(
      partial(
        pyproj.transform,
        pyproj.Proj(init='EPSG:4326'),
        pyproj.Proj(
          proj='aea',
          lat1=polygon.bounds[1],
          lat2=polygon.bounds[3])),
      polygon)
    return polygon_area.area*1e-6

  #
  def make_points(self, row):
    polygon=sg.Polygon(ast.literal_eval(row['polygon']))
    xmin,ymin,xmax,ymax=polygon.bounds
    area = self.area_from_coordinates(polygon)
    n_points=int(max(row['population']/self.max_pop, area/self.max_area)+1)
    p_id_stub='p_'+str(row['tile_id'])+'_'
    if n_points==1:
      p_id=p_id_stub+'0'
      self.point_tuples+=\
        [
          (
            p_id,
            polygon.centroid.xy[0][0],
            polygon.centroid.xy[1][0],
            area,
            row['population'],
            row['county'],
            row['p_geo_id'],
            n_points
          )
        ]
    else:
      for i in range(n_points):
        p_id=p_id_stub+str(i+1)
        self.point_tuples.\
          append(
          (
            p_id,
            xmin+random.random()*(xmax-xmin),
            ymin+random.random()*(ymax-ymin),
            1.0*area/n_points,
            1.0*row['population']/n_points,
            row['county'],
            row['p_geo_id'],
            n_points
          )
        )

  # Initialize the shape table
  # @return [DataFrame]
  #
  def init_shp_table(self):
    tuples = []
    shp = ps.open(shp_file)
    for t in shp:
      tuples.append((t.id, t.vertices.__repr__()))
    shp.close()
    return pd.DataFrame(tuples, columns=['tile_id', 'polygon'])

  # Initialize the DataBase table
  # @return [DataFrame]
  #
  def init_dbf_table(self):
    tuples = []
    dbf = ps.open(dbf_file)
    for t in dbf:
      tuples.append((t[0], t[16], t[24], t[13].split('/')[0], t[8]))
    dbf.close()
    return pd.DataFrame(tuples, columns=['tile_id', 'population', 'area', 'p_geo_id', 'county'])

  # Calulates the density of people per km^2 in a county. It averages the density of
  #  people across several shapes across an entire county as defined in the shapefile.
  #
  # @param [DataFrame] ptable - table of points
  #
  # @ return [DataFrame] talbe of points + their inferred type
  def infer_point_type(self, ptable):
    areas = ptable.groupby('p_county', as_index=False).agg({'p_area': sum, 'p_population': sum})
    areas['p_density'] = areas['p_population'] / areas['p_area']
    areas['p_type'] = np.nan
    areas.loc[areas['p_density'] < 100, 'p_type'] = 'rural'
    areas.loc[(areas['p_density'] >= 100) & (areas['p_density'] < 1000), 'p_type'] = 'suburban'
    areas.loc[(areas['p_density'] > 1000), 'p_type'] = 'urban'
    ptable = ptable.merge(areas[['p_county', 'p_type']])
    return ptable[['p_id','p_long','p_lat','p_area','p_population','p_county','p_type','p_geo_id']]

  # saves the points file according to the population and area specified
  def run(self):
    dbf_table = self.init_dbf_table()
    shp_table = self.init_shp_table()
    full_table=pd.merge(dbf_table, shp_table, on='tile_id', how='inner')
    full_table=full_table[(full_table.population>0) & (full_table.area>0)]

    full_table.apply(self.make_points,axis=1)
    ptable = \
      pd.DataFrame(
        self.point_tuples,
        columns=['p_id','p_long','p_lat','p_area','p_population','p_county','p_geo_id','n_points']
      )
    ptable = self.infer_point_type(ptable)
    ptable.to_csv(
      path_traces + '/points_'+str(int(self.max_pop))+'_'+str(float(self.max_area))+'.csv',
      index=False
    )

if __name__=='__main__':
  parser=ap.ArgumentParser(description='Mark the points')
  parser.add_argument('--dbf_file',type=str,help='File with polygon information',default=dbf_file)
  parser.add_argument('--shp_file',type=str,help='File with shape coordinates',default=shp_file)
  parser.add_argument('--max_pop',type=float,help='Maximum number of people each point can be associate to',default=500)
  parser.add_argument('--max_area',type=float,help='Maximum area (in km^2) each point can be associate to',default=5.0)
  args=parser.parse_args()
  dbf_file,shp_file,max_pop,max_area=args.dbf_file,args.shp_file,args.max_pop,(args.max_area)

  points = Points(max_pop, max_area)
  points.run()