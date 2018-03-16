#! /usr/bin/env python


#Created on 20-11-2016

# author: pdifranc
# description: generate plots for revision in IEEE Transaction on Big Data

import numpy as np,pandas as pd,matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
import shapefile
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pstats

path_figures = './img/'
path_traces = './traces/'
path_profiling = './profiling/'

class Plotter():
  def __init__(self):
    self.sf_census = shapefile.Reader('./Ireland/SmallAreaConverted_GPS.shp')
    self.recs_census = self.sf_census.records()
    self.shapes_census = self.sf_census.shapes()
    self.technology = ['3g','gsm']
    self.operator = [['mno_1','-'],['mno_2','--']]
    self.resolution = [['1000','10.0'],['500','5.0'],['300','3.0'],['100','1.0']]
    self.resolution_sampled = [['5000','50.0'], ['1000','10.0'],['500','5.0'],['300','3.0'],['100','1.0']]
    self.colors = ['#ffa500','#90ec90','#008000','red']
    self.mu = {
      'mno_1': {'rural': -0.707, 'suburban': -0.751, 'urban': -0.154},
      'mno_2':{'rural': -1.004, 'suburban': -0.507, 'urban':0.2974}
    }
    self.sigma = {
      'mno_1': {'rural': 0.683, 'suburban': 0.869, 'urban': 0.838},
      'mno_2': {'rural': 1.425, 'suburban': 0.781, 'urban': 0.621}
    }
    self.area_point = { 'rural': 48526, 'suburban': 4472, 'urban': 3530 }

  def plot_census(self, county, xlimit, ylimit, savefig = False):
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    for s in range(len(self.recs_census)):
      if (self.recs_census[s][8].find(county)) > -1:
        ptchs_census = []
        pts_census = np.array(self.shapes_census[s].points)
        prt_census = self.shapes_census[s].parts
        par_census = list(prt_census) + [pts_census.shape[0]]
        for pij_shapes in xrange(len(prt_census)):
          ptchs_census.append(Polygon(pts_census[par_census[pij_shapes]:par_census[pij_shapes + 1]]))
        ax.add_collection(PatchCollection(ptchs_census, facecolor='none', edgecolor='red', linewidths=1))
    ax.set_xlim(xmin=xlimit[0], xmax=xlimit[1])
    ax.set_ylim(ymin=ylimit[0], ymax=ylimit[1])
    ax.set_xlabel('longitude [degrees]', fontsize = 18)
    ax.set_ylabel('latitude [degrees]', fontsize = 18)
    if savefig:
      fig.savefig(path_figures + 'map_'+county.lower()+'_census.eps', bbox_inches='tight')

  # Samples
  def plot_sampled(self, county, xlimit, ylimit, savefig = False):
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    for i in range(len(self.resolution_sampled)):
      fig=plt.figure(figsize=(6,6))
      ax=fig.add_subplot(1,1,1)
      population = self.resolution_sampled[i][0];area=self.resolution_sampled[i][1]
      df_points = pd.read_csv(path_traces + '/points_'+population+'_'+area+'.csv')
      df = df_points[df_points['p_county'].str.contains(county)]
      ax.scatter(
        x=df.p_long,
        y=df.p_lat,
        c='navy',
        s=5,edgecolor='navy',
        marker='o',
        label='demand cluster'
      )
      ax.set_xlim(xmin=xlimit[0], xmax=xlimit[1])
      ax.set_ylim(ymin=ylimit[0], ymax=ylimit[1])
      ax.set_xlabel('longitude [degrees]', fontsize=18)
      ax.set_ylabel('latitude [degrees]', fontsize=18)
      if savefig:
        fig.savefig(path_figures+'points_'+county.lower()+'_census_'+population+'_'+area+'.eps',bbox_inches='tight')

  def plot_cdf_rssi(self, savefig = False):
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rc('xtick',labelsize=18)
    matplotlib.rc('ytick',labelsize=18)
    for o in self.operator:
      for t in self.technology:
        print 'DOING ', o, t
        fig = plt.figure()
        ax = fig.add_subplot(111)
        df_bs = pd.read_csv(path_traces + './base_stations.csv')
        for i in range(len(self.resolution)):
          adj = \
            pd.read_csv(path_traces + '/adj_' + self.resolution[i][0] + '_' + self.resolution[i][1] + '.csv')

          adj = adj.merge(df_bs, how='left', on='bs_id')
          adj = adj[(adj['bs_operator'] == o[0]) & (adj['bs_technology'] == t)]
          df_p = pd.read_csv(path_traces + '/points_' + self.resolution[i][0] + '_' + self.resolution[i][1] + '.csv')
          df_adj = adj.groupby('p_id', as_index=False).agg({'bp_rssi': lambda x: x.max()})
          df_adj = df_p.merge(df_adj, how='left').fillna(-180)

          ecdf = sm.distributions.ECDF(df_adj['bp_rssi'].values)
          x = np.linspace(min(df_adj['bp_rssi'].values), max(df_adj['bp_rssi'].values), len(df_adj['bp_rssi']))
          y = 1 - ecdf(x)
          ax.plot(
            x, y,
            c=self.colors[i],
            label='max_population ' + self.resolution[i][0] + '\nmax_area ' + self.resolution[i][1] + ' km$^2$',
            lw=3,
            linestyle='-'
          )
        ax.legend(loc=1, prop={'size': 14})
        ax.set_xlabel('rssi [dBm]', fontsize = 32)
        ax.set_ylabel('Complementary CDF', fontsize = 32)
        ax.grid()
        ax.set_ylim(ymin=0, ymax=1)
        ax.set_xlim(xmin=-105, xmax=-55)
        ax.set_xticks([-105, -95, -85, -75, -65, -55])
        if savefig:
          fig.savefig(path_figures + o[0] + '_' + t + '_cdf_rssi.eps', bbox_inches='tight')

  def plot_cdf_sinr(self, savefig=False):
    for o in self.operator:
      for t in self.technology:
        print 'DOING ', o, t
        fig = plt.figure()
        ax = fig.add_subplot(111)
        df_bs = pd.read_csv(path_traces + './base_stations.csv')

        for i in range(len(self.resolution)):
          adj = \
            pd.read_csv(path_traces + '/adj_' + self.resolution[i][0] + '_' + self.resolution[i][1] + '.csv')\
              .merge(df_bs, how='left', on='bs_id')
          adj = adj[(adj['bs_operator'] == o[0]) & (adj['bs_technology'] == t)]
          df_p = pd.read_csv(path_traces + '/points_' + self.resolution[i][0] + '_' + self.resolution[i][1] + '.csv')

          # Each point gets the highet sinr
          df = adj.groupby('p_id', as_index=False).agg({'bp_sinr': lambda x: x.max()})
          df = df_p.merge(df, how='left').fillna(-50)

          ecdf = sm.distributions.ECDF(df['bp_sinr'].values)
          x = np.linspace(min(df['bp_sinr'].values), max(df['bp_sinr'].values), len(df['bp_sinr']))
          y = 1 - ecdf(x)
          ax.plot(
            x, y,
            c=self.colors[i],
            label='max_population ' + self.resolution[i][0] + '\nmax_area ' + self.resolution[i][1] + ' km$^2$',
            lw=3,
            linestyle='-'
          )
        ax.legend(loc=1, prop={'size': 14})
        ax.set_xlabel('sinr [dB]', fontsize = 32)
        ax.set_ylabel('Complementary CDF', fontsize = 32)
        ax.grid()
        ax.set_ylim(ymin=0, ymax=1)
        ax.set_xlim(xmin=-5, xmax=25)
        ax.set_xticks([-5, 0, 5, 10, 15, 20, 25])
        if savefig:
          fig.savefig(path_figures + o[0] + '_' + t + '_cdf_sinr.eps', bbox_inches='tight')

  def plot_demand_cdf(self, savefig=False):
    for o in self.operator:
      area = [[0, 'rural'], [2, 'suburban'], [1, 'urban']]
      for a in area:
        area_points = self.area_point[a[1]]

        print 'Log-normal parameter', self.mu[o[0]][a[1]], self.sigma[o[0]][a[1]]

        fig = plt.figure('cdf_' + o[0]);
        ax = fig.add_subplot(1, 1, 1)
        cdf = pd.read_csv(path_traces + '/cdf_' + o[0] + '_' + a[1] + '.csv')
        ax.plot(cdf.x, cdf.y, color=self.colors[a[0]], lw=4, linestyle='--')

        fitted_sample = np.random.lognormal(self.mu[o[0]][a[1]], self.sigma[o[0]][a[1]], area_points)

        ecdf = sm.distributions.ECDF(fitted_sample)
        x = np.linspace(min(fitted_sample), max(fitted_sample), len(fitted_sample))
        y = ecdf(x)
        ax.plot(x, y, color=self.colors[a[0]], lw=4, linestyle=':')
        ax.set_xscale('log')
      p1 = plt.Rectangle((0, 0), 1, 1, fc='#ffa500')
      p2 = plt.Rectangle((0, 0), 1, 1, fc='#008000')
      p3 = plt.Rectangle((0, 0), 1, 1, fc='#90ec90')
      p4 = matplotlib.lines.Line2D([], [], c='black', linestyle='--', lw=3)
      p5 = matplotlib.lines.Line2D([], [], c='black', linestyle=':', lw=3)
      ax.legend(
        [p1, p2, p3, p4, p5],
        [
          'rural\n$\mu$=' + str(self.mu[o[0]]['rural'])[:6] + ',$\sigma$=' + str(self.sigma[o[0]]['rural'])[:5],
          'suburban\n$\mu$=' + str(self.mu[o[0]]['suburban'])[:6] + ',$\sigma$=' + str(self.sigma[o[0]]['suburban'])[:5],
          'urban\n$\mu$=' + str(self.mu[o[0]]['urban'])[:6] + ',$\sigma$=' + str(self.sigma[o[0]]['urban'])[:5],
          'Measured', 'Log-Normal fit'
        ],
        loc='best',
        ncol=1,
        prop={'size': 14}
      )
      ax.set_xlim(xmin=10e-4, xmax=10e1)
      ax.grid()
      ax.set_ylabel('cdf')
      ax.set_xlabel('demand per person [Mbit/hour]')
      if savefig:
        fig.savefig(path_figures + o[0] + '_cdf_demand_area_estimates.eps', bbox_inches='tight')

  def show_and_plot_benchmarking(self, savefig=False):
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    for res in [[1000, 10],[500, 5],[300, 3],[100, 1]]:
      p = pstats.Stats(path_profiling + 'run_'+str(res[0])+'_'+str(res[1])+'.cprof')
      p.strip_dirs().sort_stats('cumulative').print_stats(20)

    adj = (2030, 2180, 2260, 2490)
    ind = np.array([0,2,4,6])
    width = 0.50  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, adj, width, color='r')

    points = (55, 62, 67, 62)
    rects2 = ax.bar(ind + width, points, width, color='y')

    ax.set_ylabel('Execution time')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(
      [
        "max_pop = 1000\nmax_area = 10 km$^2$",
        "max_pop = 500\nmax_area = 5 km$^2$",
        "max_pop = 300\nmax_area = 3 km$^2$",
        "max_pop = 100\nmax_area = 1 km$^2$"
      ],
      rotation = 0
    )
    ax.set_ylim(ymin = 0, ymax=3000)
    ax.grid()
    ax.legend((rects1[0], rects2[0]), ('Adjacency table', 'Subscriber clusters'), loc=2)
    if savefig:
      fig.savefig(path_figures + 'benchmarking.eps', bbox_inches='tight')

  def run(self):
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rc('xtick',labelsize=18);matplotlib.rc('ytick',labelsize=18)

    # Plot census shapefiles and points.
    self.plot_census('Laois', [-7.753441,-7.006559], [52.775382,53.224618], True)
    self.plot_sampled('Laois', [-7.753441,-7.006559], [52.775382,53.224618], True)
    # CDF of RSSI
    self.plot_cdf_rssi(True)
    # CDF SINR
    self.plot_cdf_sinr(True)
    # Plot demand CDF per area type
    self.plot_demand_cdf(True)
    # Plot datasets creation
    self.show_and_plot_benchmarking(True)

if __name__=='__main__':
    plotter = Plotter()
    plotter.run()

