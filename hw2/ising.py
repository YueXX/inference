#!/usr/bin/python

import itertools
import math

class IsingModel:
  def __init__(self, dim_x, dim_y, node_weights, edge_weights,
               use_zero_one_node_vals):
    self.dim_x_ = dim_x
    self.dim_y_ = dim_y
    self.node_weights_ = node_weights
    self.edge_weights_ = edge_weights
    if use_zero_one_node_vals:
      self.node_vals_ = [0, 1]
    else:
      self.node_vals_ = [-1, 1]

    assert len(node_weights) == dim_x * dim_y
    if node_weights == 1:
      assert len(edge_weights) == 0
    else:
      assert (len(edge_weights) ==
              (dim_x-1)*dim_y + dim_x*(dim_y-1))
    self.edges_ = self.connected_nodes(self.node_weights_, self.dim_x_)

    self.phi_map_ = {}  # node value tuple -> phi
    self.partition_ = 0.0
    all_possible_node_combos = itertools.product(self.node_vals_,
                                                 repeat=len(self.node_weights_))
    for node_vals in all_possible_node_combos:
      self.phi_map_[node_vals] = self.phi(node_vals, self.node_weights_,
                                          self.edge_weights_, self.edges_)
      self.partition_ += self.phi_map_[node_vals]
    # check that it's a valid distribtion
    assert abs(sum([val/self.partition_ for val in self.phi_map_.values() ]) - 1) <= 0.01

    # x - x  0  1
    # |   |
    # x - x  2  3
    # |   |
    # x - x  4  5

    # x - x - x  0  1  2
    # |   |   |
    # x - x - x  3  4  5
    # |   |   |
    # x - x - x  6  7  8
    # |   |   |
    # x - x - x  9  10 11
  def connected_nodes(self, node_weights, dim_x):
    pairs = set()
    for node_index in range(len(node_weights)):
#      if node_index - self.dim_x_ >= 0:  # up
#        pairs.add((node_index - self.dim_x_, node_index))
      if (node_index + 1) % self.dim_x_ != 0:  # right
        pairs.add((node_index, node_index + 1))
      if node_index + self.dim_x_ < len(node_weights):  # down
        pairs.add((node_index, node_index + self.dim_x_))
#      if node_index % self.dim_x_ != 0:  # left
#        pairs.add((node_index - 1, node_index))
    return sorted(pairs)

  def phi(self, node_vals, node_weights, edge_weights, edge_pairs):
    edge_weight_val = 0.0
    node_weight_val = 0.0
    for i, edge in enumerate(edge_pairs):
      edge_weight_val += edge_weights[i]*node_vals[edge[0]]*node_vals[edge[1]]
    for node_weight, node_val in zip(node_weights, node_vals):
      node_weight_val += node_weight*node_val
    return math.exp(edge_weight_val-node_weight_val)


  def probability(self, node_vals):
    return self.phi_map_[node_vals] / self.partition_

  
def drange(start, stop, step):
  r = start
  while r < stop:
    yield r
    r += step

    
# 3x3 at most
#def FindIsingEquivalent(dim_x, dim_y, node_weights, edge_weights):
def FindIsingEquivalent(boltz_model):
  assert boltz_model.dim_x_ <= 3
  assert boltz_model.dim_y_ <= 3
  # boltz and ising value pairs
  node_value_pairs = zip(itertools.product([0, 1], repeat=len(boltz_model.node_weights_)),
                         itertools.product([-1, 1], repeat=len(boltz_model.node_weights_)))
  # TODO
  MAX_WEIGHT = max([max(node_weights), max(edge_weights)]) # TODO dubious assertion
  STEP = 0.01
  PROB_CUTOFF = 0.001
  for tmp_node_weights in itertools.product(drange(0, MAX_WEIGHT+1, STEP), repeat=len(boltz_model.node_weights_)):
#    print 'tmp_node_weights', tmp_node_weights
    for tmp_edge_weights in itertools.product(drange(0, MAX_WEIGHT+1, STEP), repeat=len(edge_weights)):
#      print 'tmp_edge_weights', tmp_edge_weights
      ising_model = IsingModel(boltz_model.dim_x_, boltz_model.dim_y_, tmp_node_weights, tmp_edge_weights,
                               use_zero_one_node_vals=False)
      candidate_found = True
      for boltz_node_vals, ising_node_vals in node_value_pairs:
        b_prob = boltz_model.probability(boltz_node_vals)
        i_prob = ising_model.probability(ising_node_vals)
        if abs(b_prob-i_prob) > PROB_CUTOFF:
          candidate_found = False
          break
      if candidate_found:
        print 'candidate found'
        return ising_model
        #candidate = (tmp_node_weights, tmp_edge_weights)
        #return candidate
  print 'no candidate found'
  return None
        

if __name__ == '__main__':
  # print '2 x 3'
  # model = IsingModel(2, 3, [1, 2, 3, 4, 5, 6], [7, 6, 5, 4, 3, 2, 1],
  #                    use_zero_one_node_vals=True)
  # for node_vals in sorted(model.phi_map_.iterkeys()):
  #   phi = model.phi_map_[node_vals]
  #   prob = model.probability(node_vals)
  #   print node_vals, 'phi:', phi, ', prob:', prob

  # print '1 node, ising, u = 3'
  # model = IsingModel(1, 1, [3], [], use_zero_one_node_vals=True)
  # for node_vals in sorted(model.phi_map_.iterkeys()):
  #   phi = model.phi_map_[node_vals]
  #   prob = model.probability(node_vals)
  #   print model.partition_
  #   print node_vals, 'phi:', phi, ', prob:', prob

  dim_x = 2
  dim_y = 1
  node_weights = [1, 1]
  edge_weights = [2]
  boltz_model = IsingModel(dim_x, dim_y, node_weights, edge_weights,
                           use_zero_one_node_vals=True)
  print 'boltz model'
  print 'node weights:', boltz_model.node_weights_, 'edge_weights:', boltz_model.edge_weights_
  ising_candidate = FindIsingEquivalent(boltz_model)
  if ising_candidate:
    print 'ising candidate'
    print 'node weights:', ising_candidate.node_weights_, 'edge_weights:', ising_candidate.edge_weights_
    print 'ising probabilities'
    node_value_pairs = zip(itertools.product([0, 1], repeat=len(node_weights)),
                           itertools.product([-1, 1], repeat=len(node_weights)))
    for boltz_node_vals, ising_node_vals in node_value_pairs:
        b_prob = boltz_model.probability(boltz_node_vals)
        i_prob = ising_candidate.probability(ising_node_vals)
        print 'boltz', boltz_node_vals, ':', b_prob
        print 'ising', ising_node_vals, ':', i_prob

  
# single node case prints
# (0,) phi: 1.0 , prob: 0.952574126822
# (1,) phi: 0.0497870683679 , prob: 0.0474258731776
# This is equivalent to Z = 1 + e^(-3), which is correct!

