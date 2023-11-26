# Adapted from: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl
import numpy as np

class Grid: # Environment
  def __init__(self, width, height, start):
    self.width = width
    self.height = height
    self.i = start[0]
    self.j = start[1]

  def set(self, rewards, actions):
    # rewards should be a dict of: (i, j): r (row, col): reward
    # actions should be a dict of: (i, j): A (row, col): list of possible actions
    self.rewards = rewards
    self.actions = actions

  def set_state(self, s):
    self.i = s[0]
    self.j = s[1]

  def current_state(self):
    return (self.i, self.j)

  def is_terminal(self, s):
    return s not in self.actions

  def move(self, action):
    # check if legal move first
    if action in self.actions[(self.i, self.j)]:
      if action == 'U':
        self.i -= 1
      elif action == 'D':
        self.i += 1
      elif action == 'R':
        self.j += 1
      elif action == 'L':
        self.j -= 1
    # return a reward (if any)
    return self.rewards.get((self.i, self.j), 0)

  def undo_move(self, action):
    # these are the opposite of what U/D/L/R should normally do
    if action == 'U':
      self.i += 1
    elif action == 'D':
      self.i -= 1
    elif action == 'R':
      self.j -= 1
    elif action == 'L':
      self.j += 1
    # raise an exception if we arrive somewhere we shouldn't be
    # should never happen
    assert(self.current_state() in self.all_states())

  def game_over(self):
    # returns true if game is over, else false
    # true if we are in a state where no actions are possible
    return (self.i, self.j) not in self.actions

  def all_states(self):
    # possibly buggy but simple way to get all states
    # either a position that has possible next actions
    # or a position that yields a reward
    return set(self.actions.keys()) | set(self.rewards.keys())


def standard_grid(st_pos=(0, 0), end_pos=(3, 3), barrier_num=1 , h=5, w=5):
    # define a grid that describes the reward for arriving at each state
    # and possible actions at each state
    # the grid looks like this
    # x means you can't go there
    # s means start position
    # number means reward at that state
    # .  .  .  1
    # .  x  . -1
    # s  .  .  .
    g = Grid(h, w, st_pos)
    rewards = {end_pos: 1}
    for _ in range(barrier_num):
      while True:
        _tmp_b = (np.random.randint(0, h), np.random.randint(0, w))
        if _tmp_b not in rewards:
          rewards[_tmp_b] = -1
          break

    actions = {}
    for i in range(h):
      for j in range(w):
        if (i, j) in rewards:
          continue
        actions[(i, j)] = []
        if i == 0:
          actions[(i, j)].append('D')
        elif i == h - 1:
          actions[(i, j)].append('U')
        else:
          actions[(i, j)].append('U')
          actions[(i, j)].append('D')
        
        if j == 0:
          actions[(i, j)].append('R')
        elif j == w - 1:
          actions[(i, j)].append('L')
        else:
          actions[(i, j)].append('L')
          actions[(i, j)].append('R')
    g.set(rewards, actions)
    return g


def negative_grid(st_pos=(0, 0), end_pos=(3, 3), barrier_num=3 , h=5, w=5, step_cost=-0.1):
  # in this game we want to try to minimize the number of moves
  # so we will penalize every move
  g = standard_grid(st_pos=st_pos, end_pos=end_pos, barrier_num=barrier_num , h=h, w=w)
  for i in range(h):
    for j in range(w):
      if (i, j) not in g.rewards and (i, j) != (g.i, g.j):
        g.rewards[(i, j)] = step_cost
  return g


def print_values(V, g):
  for i in range(g.width):
    print("---------------------------")
    for j in range(g.height):
      v = V.get((i,j), 0)
      if v >= 0:
        print(" %.2f|" % v, end="")
      else:
        print("%.2f|" % v, end="") # -ve sign takes up an extra space
    print("")


def print_policy(P, g):
  for i in range(g.width):
    print("---------------------------")
    for j in range(g.height):
      a = P.get((i,j), ' ')
      print("  %s  |" % a, end="")
    print("")