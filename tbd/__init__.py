from .version import __version__

from tbd.make_points import Points
from tbd.make_adj import Adjacency
from tbd.plotter import Plotter
import argparse as ap

def run():
  main([[5000, 50.0], [1000, 10.0], [500, 5.0], [300, 3.0], [100, 1.0]])
  Plotter().run()

def main(resolutions):
  print resolutions
  for resolution in resolutions:
    population = resolution[0]
    area = resolution[1]
    Points(population, area).run()
    Adjacency(population, area).run()

if __name__=='__main__':
  parser = ap.ArgumentParser(description='IEEE Transactions on Bi Data')
  parser.add_argument('--all', type=bool, help='Generates all files and plots', default=False)
  parser.add_argument('--max_pop', type=int, help='Maximum population that a point can represent', default=1000)
  parser.add_argument('--max_area', type=float, help='Maximum area that a point can represent', default=10.0)

  args = parser.parse_args()
  all, max_pop, max_area = args.all, args.max_pop, args.max_area

  if all:
    main([[5000, 50.0], [1000, 10.0], [500, 5.0], [300, 3.0], [100, 1.0]])
    Plotter().run()
  else:
    main([[max_pop, max_area]])