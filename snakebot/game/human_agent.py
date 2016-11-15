import os
from snakebot.utils.getch import getch
from snake_game import SnakeGame

env = SnakeGame()

while True:
  env.reset()
  done = False

  while not done:
    os.system('clear')
    env.render()
    direction = getch()
    if direction == 'w': _, done = env.step(0)
    elif direction == 's': _, done = env.step(1)
    elif direction == 'a': _, done = env.step(2)
    elif direction == 'd': _, done = env.step(3)
    elif direction == 'q': break

  print 'Game over (Press Q to quit)'
  quit = getch()
  if quit == 'q': break
