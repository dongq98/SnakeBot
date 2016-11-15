# -*- coding: utf-8 -*-

from enum import Enum
from random import randrange

Cell = Enum('Cell', 'EMPTY FOOD UP DOWN LEFT RIGHT')

class Snake:
  def __init__(self, snakeLen, snakeHead):
    self.len = snakeLen
    self.head = snakeHead
    self.tail = snakeHead[0] - snakeLen + 1, snakeHead[1]  # Start with horizontal snake
    self.dir = Cell.RIGHT  # Snake direction

  def eat(self):
    self.len = self.len + 1

class Board:
  def __init__(self, boardSize):
    self.width = boardSize[0]
    self.height = boardSize[1]
    self._board = []
    for _ in xrange(self.width):
      self._board.append([Cell.EMPTY] * self.height)

  def get(self, point):
    assert point[0] >= 0 and point[0] < self.width
    assert point[1] >= 0 and point[1] < self.height
    return self._board[point[0]][point[1]]

  def set(self, point, cell):
    assert point[0] >= 0 and point[0] < self.width
    assert point[1] >= 0 and point[1] < self.height
    self._board[point[0]][point[1]] = cell

# Get next point
def nextPoint(point, direction):
  x, y = point
  if direction == Cell.UP: y = y - 1
  if direction == Cell.DOWN: y = y + 1
  if direction == Cell.LEFT: x = x - 1
  if direction == Cell.RIGHT: x = x + 1
  return x, y

class SnakeGame:
  def __init__(self, boardSize=(16, 16), snakeLen=3):
    assert snakeLen <= boardSize[0]  # Start with horizontal snake
    self.initialLen = snakeLen
    self.board = Board(boardSize)
    self.gameOver = False
    self.score = -1  # Increase by one when self.placeFood() is executed.

    snakeHead = (randrange(snakeLen - 1, boardSize[0]), randrange(0, boardSize[1]))
    self.snake = Snake(snakeLen, snakeHead)
    for i in xrange(snakeLen):
      self.board.set((self.snake.tail[0] + i, self.snake.tail[1]), Cell.RIGHT)

    self.placeFood()

  def reset(self):
    self.__init__(boardSize=(self.board.width, self.board.height), snakeLen=self.initialLen)

  def observe(self):
    state = []
    for i in xrange(self.board.width):
      state.append([])
      for j in xrange(self.board.height):
        if self.board.get((i, j)) == Cell.EMPTY:
          state[i].append(0)
        elif self.board.get((i, j)) == Cell.FOOD:
          state[i].append(0.5)
        else:
          state[i].append(1)
    return state

  def placeFood(self):
    self.score += 1

    # Randomly place food
    for tryCount in xrange(200):
      self.food = (randrange(0, self.board.width), randrange(0, self.board.height))
      if self.board.get(self.food) == Cell.EMPTY:
        break

    # If it couldn't find the proper place for food, place food at the top left.
    if tryCount == 200:
      found = False
      for i in xrange(self.board.width):
        for j in xrange(self.board.height):
          if self.board.get((i, j)) == Cell.EMPTY:
            self.food = i, j
            found = True
            break
        if found:
          break

    self.board.set(self.food, Cell.FOOD)

  # Return (reward, done)
  def step(self, direction):  # direction: ['up', 'down', 'left', 'right', 'none']
    if self.gameOver:
      return -1, True

    reward = 0

    # When input an opposite direction, keep snake direction.
    if (direction, self.snake.dir) in [(0, Cell.DOWN), (1, Cell.UP),
        (2, Cell.RIGHT), (3,Cell.LEFT)]:
      direction = 4
    
    if direction == 0: self.snake.dir = Cell.UP
    if direction == 1: self.snake.dir = Cell.DOWN
    if direction == 2: self.snake.dir = Cell.LEFT
    if direction == 3: self.snake.dir = Cell.RIGHT
    # if direction == 4 (which is 'none'): don't change the direction

    self.board.set(self.snake.head, self.snake.dir)
    snakeHeadPrev = self.snake.head
    self.snake.head = nextPoint(self.snake.head, self.snake.dir)

    # Game over (Exit board)
    if (self.snake.head[0] < 0 or self.snake.head[0] >= self.board.width or
        self.snake.head[1] < 0 or self.snake.head[1] >= self.board.height):
      self.gameOver = True
      self.snake.head = snakeHeadPrev
      return -1, True

    headCell = self.board.get(self.snake.head)
    self.board.set(self.snake.head, self.snake.dir)

    if headCell == Cell.EMPTY:
      tailDir = self.board.get(self.snake.tail)
      self.board.set(self.snake.tail, Cell.EMPTY)
      self.snake.tail = nextPoint(self.snake.tail, tailDir)
      self.board.set(self.snake.head, self.snake.dir)
    elif headCell == Cell.FOOD:
      self.board.set(self.snake.head, self.snake.dir)
      self.snake.eat()
      reward = 1
      self.placeFood()
    else:  # Game over (Collision with body)
      self.gameOver = True
      self.snake.head = snakeHeadPrev
      return -1, True

    return reward, False

  def render(self):
    output = []
    for i in xrange(self.board.height):
      for j in xrange(self.board.width):
        if self.board.get((j, i)) == Cell.EMPTY:
          output.append('.')
        elif self.board.get((j, i)) == Cell.FOOD:
          output.append('*')
        else:
          output.append('@')
      output.append('\n')
    print ''.join(output)
