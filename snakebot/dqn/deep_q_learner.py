import random
import sys
import snakebot.game.snake_game as snake_game

import tensorflow as tf

TRAIN = True

if 'eval' in sys.argv:
  TRAIN = False

# Hyperparameters

BATCH_SIZE = 16
REPLAY_MEMORY_SIZE = 100
AGENT_HISTORY_LENGTH = 4
TARGET_NETWORK_UPDATE_FREQUENCY = 50

DISCOUNT_FACTOR = .99
LEARNING_RATE = .005

INITIAL_EPSILON = 1
FINAL_EPSILON = .1
FINAL_EXPLORATION_FRAME = 5000
REPLAY_START_SIZE = 50

NUM_EPISODES = 1000

SCREEN_WIDTH = 10
SCREEN_HEIGHT = 10
NUM_ACTIONS = 4
ACTION_MEANING = ['up', 'down', 'left', 'right']

def conv2d(x, W, s):
  return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='VALID')

def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.02)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.02)
  return tf.Variable(initial, name=name)

# state: shape [None, AGENT_HISTORY_LENGTH, SCREEN_HEIGHT, SCREEN_WIDTH]
def create_q(state, weights=None):
  state = tf.transpose(state, perm=[0, 2, 3, 1])

  if weights is not None:
    w_conv1, b_conv1, w_conv2, b_conv2, w_fc1, b_fc1, w_fc2, b_fc2 = weights

  with tf.name_scope('conv1'):
    if weights is None:
      w_conv1 = weight_variable([2, 2, AGENT_HISTORY_LENGTH, 16], name='w_conv1')
      b_conv1 = bias_variable([16], name='b_conv1')
    h_conv1 = tf.nn.relu(conv2d(state, w_conv1, 1) + b_conv1)

  with tf.name_scope('conv2'):
    if weights is None:
      w_conv2 = weight_variable([2, 2, 16, 32], name='w_conv2')
      b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, w_conv2, 1) + b_conv2)

  shape = h_conv2.get_shape().as_list()
  H, W = shape[1], shape[2]
  h_conv2_flattened = tf.reshape(h_conv2, [-1, 32*H*W], name='h_conv2_flattened')

  with tf.name_scope('fc1'):
    if weights is None:
      w_fc1 = weight_variable([32*H*W, 512])
      b_fc1 = bias_variable([512])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flattened, w_fc1) + b_fc1)

  with tf.name_scope('fc2'):
    if weights is None:
      w_fc2 = weight_variable([512, NUM_ACTIONS])
      b_fc2 = bias_variable([NUM_ACTIONS])
    h_fc2 = tf.matmul(h_fc1, w_fc2) + b_fc2  # h_fc2: shape [None, NUM_ACTIONS]

  return h_fc2, (w_conv1, b_conv1, w_conv2, b_conv2, w_fc1, b_fc1, w_fc2, b_fc2)

def put_experience(s, a, r, s_next, done):
  global D_index

  new_experience = (s, a, r, s_next, done)
  if len(D) < REPLAY_MEMORY_SIZE:
    D.append(new_experience)
  else:
    D[D_index] = new_experience
    D_index += 1
    if D_index == len(D):
      D_index = 0

def main():
  with tf.Session() as sess:
    print 'Initializing...'

    env = snake_game.SnakeGame(boardSize=(SCREEN_WIDTH, SCREEN_HEIGHT))
    state = tf.placeholder(tf.float32,
        shape=[None, AGENT_HISTORY_LENGTH, SCREEN_WIDTH, SCREEN_HEIGHT], name='state')

    global D, D_index
    D = []  # Replay memory
    D_index = 0

    step = 0
    max_score = 0

    q_network, theta = create_q(state)  # Q network with random weights
    target_network, target_theta = create_q(state, theta)

    # RMSPropOptimizer is required to be defined before initializing variables.
    ph_y = tf.placeholder(tf.float32, name='y')
    ph_a = tf.placeholder(tf.int32, name='a')
    q_value = tf.reduce_sum(q_network * tf.one_hot(ph_a, NUM_ACTIONS, 1., 0.))
    loss = tf.square(ph_y - q_value)
    train_op = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss)

    sess.run(tf.initialize_all_variables())

    initial_observation_done = False
    
    for episode in xrange(NUM_EPISODES):
      env.reset()
      done = False
      observation = env.observe()
      current_state = [observation] * AGENT_HISTORY_LENGTH

      epsilon = INITIAL_EPSILON

      while not done:
        if step < FINAL_EXPLORATION_FRAME:
          epsilon = INITIAL_EPSILON - step * (INITIAL_EPSILON-FINAL_EPSILON) / FINAL_EXPLORATION_FRAME
        else:
          epsilon = FINAL_EPSILON

        # A uniform random policy is run for some initial frames.
        if len(D) < REPLAY_START_SIZE:
          action = random.randrange(NUM_ACTIONS)
          reward, done = env.step(action)
          step += 1
          new_state = current_state[1:] + [env.observe()]
          put_experience(current_state, action, reward, new_state, done)

        else:
          if not initial_observation_done:
            print 'Memory replay start'
            initial_observation_done = True

          random_action = False
          # With probability epsilon select a random action.
          if random.random() < epsilon:
            action = random.randrange(NUM_ACTIONS)
            random_action = True
          # Otherwise select argmax of current Q network.
          else:
            action = sess.run(tf.argmax(q_network, 1), feed_dict={state: [current_state]})[0]

          reward, done = env.step(action)
          step += 1
          new_state = current_state[1:] + [env.observe()]
          put_experience(current_state, action, reward, new_state, done)

          minibatch = random.sample(D, BATCH_SIZE)

          for s, a, r, s_next, d in minibatch:
            if d:
              y = r
            else:
              y = r + DISCOUNT_FACTOR * sess.run(
                  tf.reduce_max(target_network, reduction_indices=1), feed_dict={state: [s_next]})[0]

            _, current_loss = sess.run([train_op, loss], feed_dict={ph_y: y, ph_a: a, state: [s]})

          if step % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
            target_network, target_network_theta = create_q(state, theta)
          
          max_score = env.score if max_score < env.score else max_score

          print '┌ Episode: ' + str(episode) + '\tStep: ' + str(step) + '\tEpsilon:' + str(epsilon)
          print '│ Score: ' + str(env.score) + '\tAction: ' + ACTION_MEANING[action] + \
                  '\tRandom action: ' + str(random_action)
          print '└ Max score: ' + str(max_score)

      if initial_observation_done:
        print 'Episode ' + str(episode) + ' done\n'

if __name__ == '__main__':
  main()