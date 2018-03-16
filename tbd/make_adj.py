#! /usr/bin/env python

# author: pdifranc
# date: 06/09/2016
# description: mobile network planning adjacency file IEEE Transaction of Big Data

import numpy as np, pandas as pd
from tqdm import tqdm

import argparse as ap

all_sectors = set()
path_traces = './traces'

# Class to construct the adjacency file according to the population and area
#  specified.
class Adjacency():
  def __init__(self, population, area):
    self.population = population
    self.area = area
    self.points = pd.read_csv(path_traces + '/points_' + str(population) + '_' + str(area) + '.csv')
    self.bs = pd.read_csv(path_traces + '/base_stations.csv')
    self.adj = pd.DataFrame()                     # final dataframe
    self.types = ['rural', 'urban', 'suburban']   # list of point types, it defines the propagation madel to use
    self.sensitivity = -105                       # dBm
    self.H_b = 30
    self.H_m = 1.5
    self.No_dbm = (-104)  # Thermal noise [dBm]
    self.No_lin = np.power(10, .1 * (self.No_dbm))
    self.frequencies = self.bs.bs_fc.unique().tolist()
    self.technologies = self.bs.bs_technology.unique().tolist()
    self.operators = self.bs.bs_operator.unique().tolist()

  # Calculates the distance between a point (lat1, lon1) and an
  #  array of coordinates (lat2 ,lon2)
  # @param [Float] lon1 - longitude of reference point
  # @param [Float] lat1 - latitude of reference point
  # @param [Array<Float>] lon2 - longitute of a series of points
  # @param [Array<Float>] lat2 - Latitude of a series of points
  #
  # @return [Array<Float>] distance in km from the points of references and all other points.
  def haversine(self, lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1,lat1, lon2, lat2 = np.radians(lon1), np.radians(lat1), np.radians(lon2), np.radians(lat2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

  # Calculates all distances from the base station to the points given then,
  #  based on the maximum path loss, asses if a point is covered or not.
  #
  # @params[DataFrame] points
  # @params[Float] tx_lat - base station latitude
  # @params[Float] tx_long - base station longitude
  # @params[Float] tx_power - base station power [in dBm]
  # @params[Integer] frequency - base station operative frequency
  # @params[String] terrain - type of points considered
  #
  # @return[Array] list of points covered
  def find_points_covered(self, points, tx_lat, tx_long, tx_power, frequency, terrain):
    a_H_m = (1.1 * np.log10(frequency) - 0.7) * self.H_m - (1.56 * np.log10(frequency) - 0.8)
    index = np.arange(len(points)).reshape((len(points), 1))  # To retrive the index
    D = np.c_[self.haversine(tx_long, tx_lat, points.p_long.values, points.p_lat.values), index]
    max_L = tx_power - self.sensitivity
    # Using the Extended Hata model
    if (frequency >= 150) and (frequency <= 1500):
      if terrain == 'urban':
        max_d = np.power(
          10,
          (max_L + a_H_m - (69.6 + 26.2 * np.log10(frequency) - 13.82 * np.log10(self.H_b))) /
            (44.9 - 6.55 * np.log10(self.H_b))
        )
      elif terrain == 'suburban':
        correction_factor = -2 * np.power(np.log10(frequency / 28), 2) - 5.4
        max_d = np.power(
          10,
          (max_L - correction_factor + a_H_m - (69.6 + 26.2 * np.log10(frequency) - 13.82 * np.log10(self.H_b))) /
            (44.9 - 6.55 * np.log10(self.H_b))
        )
      elif terrain == 'rural':
        correction_factor = -4.78 * np.power(np.log10(frequency), 2) + 18.33 * np.log10(frequency) - 40.94
        max_d = np.power(
          10,
          (max_L - correction_factor + a_H_m - (69.6 + 26.2 * np.log10(frequency) - 13.82 * np.log10(self.H_b))) /
            (44.9 - 6.55 * np.log10(self.H_b))
        )
    elif (frequency > 1500) and (frequency <= 2000):
      if terrain == 'urban':
        max_d = np.power(
          10,
          (max_L + a_H_m - (46.3 + 33.9 * np.log10(frequency) - 13.82 * np.log10(self.H_b))) /
            (44.9 - 6.55 * np.log10(self.H_b))
        )
      elif terrain == 'suburban':
        correction_factor = -2 * np.power(np.log10(frequency / 28), 2) - 5.4
        max_d = np.power(
          10,
          (max_L - correction_factor + a_H_m - (46.3 + 33.9 * np.log10(frequency) - 13.82 * np.log10(self.H_b))) /
            (44.9 - 6.55 * np.log10(self.H_b))
        )
      elif terrain == 'rural':
        correction_factor = -4.78 * np.power(np.log10(frequency), 2) + 18.33 * np.log10(frequency) - 40.94
        max_d = np.power(
          10,
          (max_L - correction_factor + a_H_m - (46.3 + 33.9 * np.log10(frequency) - 13.82 * np.log10(self.H_b))) /
            (44.9 - 6.55 * np.log10(self.H_b))
        )
    elif (frequency > 2000) and (frequency <= 3000):
      if terrain == 'urban':
        max_d = np.power(
          10,
          (max_L + a_H_m - (46.3 + 33.9 * np.log10(2000) + 10 * np.log10(frequency / 2000) - 13.82 * np.log10(self.H_b))) /
            (44.9 - 6.55 * np.log10(self.H_b))
        )
      elif terrain == 'suburban':
        correction_factor = -2 * np.power(np.log10(2000 / 28), 2) - 5.4
        max_d = np.power(
          10,
          (max_L - correction_factor + a_H_m - (46.3 + 33.9 * np.log10(2000) + 10 * np.log10(frequency / 2000) - 13.82 * np.log10(self.H_b))) /
            (44.9 - 6.55 * np.log10(self.H_b))
        )
      elif terrain == 'rural':
        correction_factor = -4.78 * np.power(np.log10(2000), 2) + 18.33 * np.log10(2000) - 40.94
        max_d = np.power(
          10,
          (max_L - correction_factor + a_H_m - (46.3 + 33.9 * np.log10(2000) + 10 * np.log10(frequency / 2000) - 13.82 * np.log10(self.H_b))) /
            (44.9 - 6.55 * np.log10(self.H_b))
        )
    return D[D[:, 0] < max_d]

  # Compute the RSSI on each point/base station pair identified. It first
  #  calculates the attenuation, then the recieved power at the point from the
  #  base station.
  #
  # @params[DataFrame] adj - adjacency table with legitimate pairs
  #
  # @returns[DataFrame]
  #
  def compute_rssi(self, adj):
    alpha = 1
    a_Hm = (1.1 * np.log10(adj['bs_fc']) - 0.7) * 1.5 - (1.56 * np.log10(adj['bs_fc']) - 0.8) + 0
    b_Hb = 0
    # Compute the Attenuation
    adj['bp_attenuation'] = np.NAN
    # f<1500
    # Open area
    adj.loc[(adj['p_type'] == 'rural') & (adj['bs_fc'] < 1500), 'bp_attenuation'] = \
      69.6 + 26.2 * np.log10(
        adj['bs_fc'][(adj['p_type'] == 'rural') & (adj['bs_fc'] < 1500)]
      ) - \
      13.82 * np.log10(30) + \
      (44.9 - 6.55 * np.log10(30)) * \
      np.power(
        (np.log10(adj['bp_distance'][(adj['p_type'] == 'rural') & (adj['bs_fc'] < 1500)])),
        alpha
      ) - \
      a_Hm[
        (adj['p_type'] == 'rural') &
        (adj['bs_fc'] < 1500)
      ] - \
      b_Hb - \
      4.78 * np.power(
        np.log10(adj['bs_fc'][(adj['p_type'] == 'rural') & (adj['bs_fc'] < 1500)]),
        2
      ) + 18.33 * np.log10(adj['bs_fc'][(adj['p_type'] == 'rural') & (adj['bs_fc'] < 1500)]) - \
      40.94
    # Urban
    adj.loc[(adj['p_type'] == 'urban') & (adj['bs_fc'] < 1500), 'bp_attenuation'] = \
      69.6 + 26.2 * np.log10(
        adj['bs_fc'][(adj['p_type'] == 'urban') & (adj['bs_fc'] < 1500)]
      ) - \
      13.82 * np.log10(30) + \
      (44.9 - 6.55 * np.log10(30)) * \
      np.power(
        (np.log10(adj['bp_distance'][(adj['p_type'] == 'urban') & (adj['bs_fc'] < 1500)])),
        alpha
      ) - \
      a_Hm[
        (adj['p_type'] == 'urban') &
        (adj['bs_fc'] < 1500)
      ] - \
      b_Hb
    # Suburban
    adj.loc[(adj['p_type'] == 'suburban') & (adj['bs_fc'] < 1500), 'bp_attenuation'] = \
      69.6 + 26.2 * np.log10(
        adj['bs_fc'][(adj['p_type'] == 'suburban') & (adj['bs_fc'] < 1500)]
      ) - \
      13.82 * np.log10(30) + \
      (44.9 - 6.55 * np.log10(30)) * \
      np.power(
        (np.log10(adj['bp_distance'][(adj['p_type'] == 'suburban') & (adj['bs_fc'] < 1500)])),
        alpha
      ) - \
      a_Hm[
        (adj['p_type'] == 'suburban') &
        (adj['bs_fc'] < 1500)
      ] - \
      b_Hb - \
      2 * np.power(
        np.log10(adj['bs_fc'][(adj['p_type'] == 'suburban') & (adj['bs_fc'] < 1500)]) / 28,
        2
      ) - 5.4
    # 1500<f<2000
    # Open Area
    adj.loc[(adj['p_type'] == 'rural') & (adj['bs_fc'] > 1500) & (adj['bs_fc'] <= 2000), 'bp_attenuation'] = \
      46.3 + 33.9 * np.log10(
        adj['bs_fc'][(adj['p_type'] == 'rural') & (adj['bs_fc'] > 1500) & (adj['bs_fc'] <= 2000)]
      ) - \
      13.82 * np.log10(30) + \
      (44.9 - 6.55 * np.log10(30)) * \
      np.power(
        (np.log10(adj['bp_distance'][(adj['p_type'] == 'rural') & (adj['bs_fc'] > 1500) & (adj['bs_fc'] <= 2000)])),
        alpha
      ) - \
      a_Hm[
        (adj['p_type'] == 'rural') &
        (adj['bs_fc'] > 1500) &
        (adj['bs_fc'] <= 2000)
      ] - \
      b_Hb - \
      4.78 * np.power(
        np.log10(adj['bs_fc'][(adj['p_type'] == 'rural') & (adj['bs_fc'] > 1500) & (adj['bs_fc'] <= 2000)]),
        2
      ) + \
      18.33 * np.log10(
        adj['bs_fc'][(adj['p_type'] == 'rural') & (adj['bs_fc'] > 1500) & (adj['bs_fc'] <= 2000)]
      ) - \
      40.94
    # Urban
    adj.loc[(adj['p_type'] == 'urban') & (adj['bs_fc'] > 1500) & (adj['bs_fc'] <= 2000), 'bp_attenuation'] = \
      46.3 + 33.9 * np.log10(
        adj['bs_fc'][(adj['p_type'] == 'urban') & (adj['bs_fc'] > 1500) & (adj['bs_fc'] <= 2000)]
      ) - \
      13.82 * np.log10(30) + \
      (44.9 - 6.55 * np.log10(30)) * \
      np.power(
        (np.log10(adj['bp_distance'][(adj['p_type'] == 'urban') & (adj['bs_fc'] > 1500) & (adj['bs_fc'] <= 2000)])),
        alpha
      ) - \
      a_Hm[
        (adj['p_type'] == 'urban') &
        (adj['bs_fc'] > 1500) &
        (adj['bs_fc'] <= 2000)
      ] - \
      b_Hb
    # suburban
    adj.loc[(adj['p_type'] == 'suburban') & (adj['bs_fc'] > 1500) & (adj['bs_fc'] <= 2000), 'bp_attenuation'] = \
      46.3 + 33.9 * np.log10(
        adj['bs_fc'][(adj['p_type'] == 'suburban') & (adj['bs_fc'] > 1500) & (adj['bs_fc'] <= 2000)]
      ) - \
      13.82 * np.log10(30) + \
      (44.9 - 6.55 * np.log10(30)) * \
      np.power(
        (np.log10(adj['bp_distance'][(adj['p_type'] == 'suburban') & (adj['bs_fc'] > 1500) & (adj['bs_fc'] <= 2000)])),
        alpha
      ) - \
      a_Hm[
        (adj['p_type'] == 'suburban') &
        (adj['bs_fc'] > 1500) &
        (adj['bs_fc'] <= 2000)
      ] - \
      b_Hb - \
      2 * np.power(
        np.log10(adj['bs_fc'][(adj['p_type'] == 'suburban') & (adj['bs_fc'] > 1500) & (adj['bs_fc'] <= 2000)]) / 28,
        2
      ) - \
      5.4

    adj['bp_rssi'] = np.power(10, .1 * (adj['bs_power'] - adj['bp_attenuation']))
    return adj

  # Convertes the SINR from linear to dB
  #
  # @return[DataFrame]
  #
  def add_sinr(self, adj):
    adj['bp_sinr'] = 10 * np.log10(adj['bp_rssi'] / (adj['all_rx'] - adj['bp_rssi'] + self.No_lin))
    return adj

  # Computes the Signal to Interfere + Noise Ratio. The assumption is that
  #  different operators do not interfere with each other (in practice,
  #  regulators assign adjacent portions of the bandwidth reserved to a technology
  #  i.e. 3G, GSM, to different operators to avoid mutual interference).
  #
  # @return[DataFrame]
  #
  def compute_sinr(self):
    adj = pd.DataFrame()
    for tech in self.technologies:
      for fc in self.frequencies:
        for operator in self.operators:
          adj_mno = self.adj.query(
            'bs_operator=="'+operator+'" and bs_technology=="' + tech + '" and bs_fc==' + str(fc)
          )
          if len(adj_mno) != 0:
            adj_mno = adj_mno.merge(
              (adj_mno.groupby('p_id').agg({'bp_rssi': lambda x: x.sum()}).reset_index()).rename(
                columns={'bp_rssi': 'all_rx'}), on='p_id'
            )
            adj_mno = self.add_sinr(adj_mno)
            adj = adj.append(adj_mno, ignore_index=True)
    # convert the rssi in dBm
    adj['bp_rssi'] = 10 * np.log10(adj['bp_rssi'])
    return adj[['bs_id', 'p_id', 'bp_distance', 'bp_attenuation', 'bp_rssi', 'bp_sinr']]

  def compute_network_params(self, adj):
    adj = self.compute_rssi(adj)
    return adj

  def create_adj(self, points, p, bs_id, p_type, bs_power, frequency, operator, technology):
    adj = pd.DataFrame()
    adj['p_id'] = points['p_id'][p[:, 1]].values
    adj['bs_id'] = bs_id
    adj['bs_power'] = bs_power
    adj['bs_fc'] = frequency
    adj['bp_distance'] = p[:, 0]
    adj['bs_operator'] = operator
    adj['p_type'] = p_type
    adj['bs_technology'] = technology
    adj = self.compute_network_params(adj)
    return adj

  def run(self):
    df_list = []
    for index, row in tqdm(self.bs.iterrows(), total=self.bs.shape[0]):
      for point_type in self.types:
        bs_id = row.bs_id
        bs_lat, bs_long, bs_power = row.bs_lat, row.bs_long, row.bs_power
        bs_fc, bs_operator = row.bs_fc, row.bs_operator
        bs_technology = row.bs_technology
        points = self.points[self.points['p_type'] == point_type].reset_index(drop=True)
        points_covered = \
          self.find_points_covered(
            points,
            bs_lat,
            bs_long,
            bs_power,
            bs_fc,
            point_type
          )
        adj = \
          self.create_adj(
            points,
            points_covered,
            bs_id,
            point_type,
            bs_power,
            bs_fc,
            bs_operator,
            bs_technology
          )
        df_list.append(adj)
    self.adj = pd.concat(df_list, ignore_index=True)
    self.adj = self.compute_sinr()
    self.adj.to_csv(
      path_traces + '/adj_' + str(self.population) + '_' + str(self.area) + '.csv',
      index=False
    )

if __name__ == '__main__':
  parser = ap.ArgumentParser(description='Make the adj table (no demand)')
  parser.add_argument(
    '--population',
    type=float,
    help='Maximum number of people each point can be associate to',
    default=1000
  )
  parser.add_argument(
    '--area',
    type=float,
    help='Maximum area (in km^2) each point can be associate to',
    default=10.0
  )
  args = parser.parse_args()
  obj = Adjacency(int(args.population), float(args.area))
  obj.run()