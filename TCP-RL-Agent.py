#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import sys
import argparse
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#import keras
import tensorflow as tf
from keras.callbacks import History 
import csv
#from livelossplot import PlotLossesKeras,PlotLossesKerasTF
from keras_lr_finder import LRFinder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import time
import timeit
from ns3gym import ns3env
from tcp_base import TcpTimeBased, TcpEventBased

def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'mse' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'mse' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('mse')
    plt.xlabel('Epochs')
    plt.ylabel('mse')
    plt.legend()
    plt.show()

try:
	w_file = open('run.log', 'w')
except:
	w_file = sys.stdout
parser = argparse.ArgumentParser(description='Start simulation script on/off')
parser.add_argument('--start',
					type=int,
					default=1,
					help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations',
					type=int,
					default=1,
					help='Number of iterations, Default: 1')
parser.add_argument('--steps',
					type=int,
					default=100,
					help='Number of steps, Default 100')
args = parser.parse_args()

#pltlosses=PlotLossesKeras()
startSim = bool(args.start)
iterationNum = int(args.iterations)
maxSteps = int(args.steps)

port = 5555
simTime = maxSteps / 10.0 # seconds
seed = 12
simArgs = {"--duration": simTime,}

dashes = "-"*18
input("[{}Press Enter to start{}]".format(dashes, dashes))

# create environment
env = ns3env.Ns3Env(port=port, startSim=startSim, simSeed=seed, simArgs=simArgs)

ob_space = env.observation_space
ac_space = env.action_space

# TODO: right now, the next action is selected inside the loop, rather than using get_action.
# this is because we use the decaying epsilon-greedy algo which needs to use the live model
# somehow change or put that logic in an `RLTCP` class that inherits from the Tcp class, like in tcp_base.py,
# then move the class to tcp_base.py and use that agent here
def get_agent(state):
	socketUuid = state[0]
	tcpEnvType = state[1]
	tcpAgent = get_agent.tcpAgents.get(socketUuid, None)
	if tcpAgent is None:
		# get a new agent based on the selected env type
		if tcpEnvType == 0:
			# event-based = 0
			tcpAgent = TcpEventBased()
		else:
			# time-based = 1
			tcpAgent = TcpTimeBased()
		tcpAgent.set_spaces(get_agent.ob_space, get_agent.ac_space)
		get_agent.tcpAgents[socketUuid] = tcpAgent

	return tcpAgent

# initialize agent variables
# (useless until the above todo is fixed)
get_agent.tcpAgents = {}
get_agent.ob_space = ob_space
get_agent.ac_space = ac_space
def modelerDP(input_size, output_size):
	"""
	Designs a fully connected neural network.
	"""
	model = tf.keras.Sequential()

	# input layer
	model.add(tf.keras.layers.Dense((input_size + output_size) // 2, input_shape=(input_size,), activation='relu'))

	model.add(tf.keras.layers.Dense(100, activation='relu'))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dense(100, activation='relu'))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dense(output_size,activation='softmax'))

    
    
	model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2),loss='categorical_crossentropy',metrics=['mse'])
	
	return model
def modelerDPM(input_size, output_size):
    """
    Designs a fully connected neural network with 10 layers.
    """
    model = tf.keras.Sequential()

    # Input layer
    model.add(tf.keras.layers.Dense((input_size + output_size) // 2, input_shape=(input_size,), activation='relu'))

    # Hidden layers
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    # Output layer
    model.add(tf.keras.layers.Dense(output_size, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2), loss='categorical_crossentropy', metrics=['mse'])

    return model

def modeler(input_size, output_size):
	"""
	Designs a fully connected neural network.
	"""
	model = tf.keras.Sequential()

	# input layer
	model.add(tf.keras.layers.Dense((input_size + output_size) // 2, input_shape=(input_size,), activation='relu'))

	# output layer
	# maps previous layer of input_size units to output_size units
	# this is a classifier network
	model.add(tf.keras.layers.Dense(output_size, activation='softmax'))
	#model.add(tf.keras.layers.Dense(output_size, activation='sigmoid'))
	return model

state_size = ob_space.shape[0] - 4 # ignoring 4 env attributes

action_size = 100
action_mapping = {}  # Inicialize o dicionário vazio


random.seed(42)
action_size = 100
action_mapping = {}

for i in range(action_size):
     action_mapping[i] = random.randint(-100, 100)


# build model
#model = modeler(state_size, action_size)
#model.compile(
#	optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2),
#	loss='categorical_crossentropy',
	
#	metrics=['mse']
#)
model = modelerDPM(state_size, action_size)
#model.summary()
# initialize decaying epsilon-greedy algorithm
# fine-tune to ensure balance of exploration and exploitation
epsilon = 0.9
epsilon_decay_param = iterationNum * 2
min_epsilon = 0.1
epsilon_decay = (((epsilon_decay_param*maxSteps) - 1.0) / (epsilon_decay_param*maxSteps))

# initialize Q-learning's discount factor
discount_factor = 0.95

rewardsum = 0
rew_history = []
cWnd_history = []
pred_cWnd_history = []
rtt_history = []
tp_history = []
loss_history=[]
recency = maxSteps // 15
accuracy=0
cont=0
conv=0
maxRewconv=0
done = False
mse_histori=[]
pretty_slash = ['\\', '|', '/', '-']
inicio = timeit.default_timer()
for iteration in range(iterationNum):
	
	# set initial state
	state = env.reset()
	# ignore env attributes: socketID, env type, sim time, nodeID
	state = state[4:]

	cWnd = state[1]
	init_cWnd = cWnd
	#print(cWnd)
	state = np.reshape(state, [1, state_size])
	try:
		for step in range(maxSteps):
			pretty_index = step % 4
			print("\r{}\r[{}] Logging to file {} {}".format(
				' '*(25+len(w_file.name)),
				pretty_slash[pretty_index],
				w_file.name,
				'.'*(pretty_index+1)
			), end='')

			print("[+] Step: {}".format(step+1), file=w_file)

			# Epsilon-greedy selection
			if step == 0 or np.random.rand(1) < epsilon:
				# explore new situation
				action_index = np.random.randint(0, action_size)
				print("\t[*] Random exploration. Selected action: {}".format(action_index), file=w_file)
			else:
				# exploit gained knowledge
				action_index = np.argmax(model.predict(state)[0])
				print("\t[*] Exploiting gained knowledge. Selected action: {}".format(action_index), file=w_file)

			# Calculate action
			calc_cWnd = cWnd + action_mapping[action_index]

			# Config 1: no cap
			# new_cWnd = calc_cWnd

			# Config 2: cap cWnd by half upon congestion
			# ssThresh is set to half of cWnd when congestion occurs
			# prevent new_cWnd from falling too low
			# ssThresh = state[0][0]
			# new_cWnd = max(init_cWnd, (min(ssThresh, calc_cWnd)))

			# Config 3: if throughput cap detected, fix cWnd
			# detect cap by checking if recent variance less than 1% of current 
			thresh = state[0][0] # ssThresh
			if step+1 > recency:
				tp_dev = math.sqrt(np.var(tp_history[(-recency):]))
				tp_1per = 0.01 * throughput
				if tp_dev < tp_1per:
					thresh = cWnd
			new_cWnd = max(init_cWnd, (min(thresh, calc_cWnd)))
			
			# Config 4: detect throughput cap by checking against experimentally determined value
			# thresh = state[0][0] # ssThresh
			# if step+1 > recency:
			# 	if throughput > 216000: # must be tuned based on bandwidth
			# 		thresh = cWnd
			# new_cWnd = max(init_cWnd, (min(thresh, calc_cWnd)))

			new_ssThresh = int(cWnd/2)
			actions = [new_ssThresh, new_cWnd]

			# Take action step on environment and get feedback
			#next_state, reward, done, _ = env.step(actions)
			env.ns3ZmqBridge.step(actions)
			next_state=env.ns3ZmqBridge.get_obs()
			reward=env.ns3ZmqBridge.get_reward()
			done=env.ns3ZmqBridge.is_game_over()
			rewardsum += reward

			next_state = next_state[4:]
			cWnd = next_state[1]
			rtt = next_state[7]
			throughput = next_state[11]
			
			print("\t[#] Next state: ", next_state, file=w_file)
			print("\t[!] Reward: ", reward, file=w_file)

			next_state = np.reshape(next_state, [1, state_size])
			

			# Train incrementally
			# DQN - function approximation using neural networks
			target = reward
			#target = 1
			#reward=1
			
			if not done:
				cont+=1
				print("\n----------------------------------------")
				target = (reward + discount_factor * np.amax(model.predict(next_state)[0]))
				print(target)
			target_f = model.predict(state)
			target_f[0][action_index] = target
			#atual=model.predict(state)
			#sprint(state,target_f)
			history = History()
			#model.fit(state, target_f , epochs=30, callbacks=[pltlosses],verbose=False)
			#accuracy+=history.history['accuracy']	
			model.fit(state, target_f , epochs=30,verbose=False,callbacks=[history])
			#model.history['accuracy']
			print("\n########3")
			#print(history.history['mse'])
			mse_histori.append(history.history['mse'])
			loss_history.append(history.history['loss'])
			# Update state
			state = next_state


			if done:
				print("[X] Stopping: step: {}, reward sum: {}, epsilon: {:.2}"
						.format(step+1, rewardsum, epsilon),
						file=w_file)
				break

			if epsilon > min_epsilon:
				epsilon *= epsilon_decay
			if step >0:
				temp=max(rew_history)
			else:
				temp=0
			# Record information
			rew_history.append(rewardsum)
			rtt_history.append(rtt)
			cWnd_history.append(cWnd)
			tp_history.append(throughput)
			
			if temp == max(rew_history):
				cont += 1
				if cont == 10:
					print("reward convergiu no step " + str(step))
					maxRewconv=temp
					conv=step
			else:
				cont = 0
			
		print("\n[O] Iteration over.", file=w_file)
		print("[-] Final epsilon value: ", epsilon, file=w_file)
		print("[-] Final reward sum: ", rewardsum, file=w_file)
		print("[-] Final mse_histori: ", mse_histori, file=w_file)
		print("\n[-] Final loss:\n ", loss_history, file=w_file)
		#print(mse_histori)
		print()
		#print(history.history['loss'])
		
		
	finally:
		print()
		if iteration+1 == iterationNum:
			break
		# if str(input("[?] Continue to next iteration? [Y/n]: ") or "Y").lower() != "y":



		# 	break
#
# 
# 
#fim = timeit.default_timer()
#print ('duracao: %f' % (fim - inicio))
temp=loss_history[-1]
plt.figure(figsize=(20, 10))
plt.plot(range(len(temp)), temp, marker="", linestyle="-")
plt.title('loss_histori')
plt.xlabel('Steps')
plt.ylabel('loss')
plt.savefig('loss_plot_Daniel.pdf')
#plt.show()

plt.figure(figsize=(20, 10))
plt.plot(range(maxSteps), tp_history, marker="", linestyle="-")
plt.title('Throughput over time')
plt.xlabel('Steps')
plt.ylabel('Throughput (bits)')
plt.savefig('throughput_plot_Daniel.pdf')
#plt.show()
print('\n max Throughput: '+str(max(tp_history)))
#print(tp_history)

plt.figure(figsize=(20, 10))
plt.plot(range(maxSteps), rew_history, marker="", linestyle="-")
plt.title('Reward sum plot')
plt.xlabel('Steps')
plt.ylabel('Accumulated reward')
plt.savefig('reward_plot_Daniel.pdf')
print('\n max reward: '+str(max(rew_history))) 
#print(rew_history)
#plt.show()

plt.figure(figsize=(20, 10))
plt.plot(range(len(mse_histori[-1])), mse_histori[-1], marker="", linestyle="-")
plt.title('mse_histori')
plt.xlabel('Steps')
plt.ylabel('mse')
plt.savefig('mse_plot_Daniel.pdf')



fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
		
ax1.plot(range(maxSteps),rew_history, label='reward')
ax1.set_ylabel('reward', color='tab:blue')
#ax1.set_ylabel('Convergence Step',conv)
print('\nconvergiu no step '+str(conv))
ax2.plot(range(maxSteps), rew_history, marker="", linestyle="-", label='Rewards')
ax2.plot(range(len(cWnd_history)), cWnd_history, marker="", linestyle="-", label='Cwnd')
ax2.set_ylabel('Rewards and Cwnd', color='tab:orange')

plt.legend(loc='upper left')
plt.title('Gráficos Múltiplos')
#ax1.annotate('max', xy=(conv-5,maxRewconv), xytext=(conv, 38),
            #arrowprops=dict(facecolor='black', shrink=0.05))

plt.savefig('conv_Daniel2.pdf')
#plt.show()
#print(mse_histori)
# print(loss_history)
#eixo_x = np.linspace(0, 30, len(loss_history[0]))
#print('eixo y'+str(eixo_x))
#plot_history(history)
#for vet_step in loss_history:
	#print(str(vet_step)+'\n')

	#plt.plot(eixo_x, vet_step )

#plt.xlabel("Eixo X")
#plt.ylabel("Valores")
#plt.title("Gráfico dos Subvetores")
#plt.show()
#fig, ax1 = plt.subplots(layout='constrained')

# Configure os subvetores
"""
for i, subvetor in enumerate(loss_history):
    color = 'C{}'.format(i)  # Escolhe uma cor diferente para cada subvetor
    label = f'Step {i + 1}'
    
    if i == 0:
        ax = ax1
    else:
        ax = ax1.twinx()
    
    ax.plot(eixo_x, subvetor, color, label=label)
    ax.set_ylabel(f'Valores {label} ', color=color)
    ax.tick_params('y', colors=color)

# Configurações do gráfico
#plt.figure(figsize=(20, 20))
plt.title('Gráfico dos Subvetores2')
lines, labels = ax1.get_legend_handles_labels()
plt.legend(lines, labels, loc='upper left')
# Set Axes
plt.xlim(0, 30)
plt.ylim(0, 9)
#plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('loss_history.pdf')
#plt.show()
print(len(loss_history[0]))
temp=loss_history[-1]
print(temp)


"""






"""

plot_history(history)
csv_file = "loss_history.csv"
flat_loss_history = [value for sublist in loss_history for value in sublist]
with open(csv_file, 'w', newline='') as file:
	writer = csv.writer(file)
	writer.writerow(flat_loss_history)
mpl.rcdefaults()
mpl.rcParams.update({'font.size': 16})

# Gráfico 1: Congestion windows
plt.figure(figsize=(20, 10))
plt.plot(range(maxSteps), cWnd_history, marker="", linestyle="-")
plt.title('Congestion windows')
plt.xlabel('Steps')
plt.ylabel('CWND (segments)')
plt.savefig('cwnd_plot_Daniel.pdf')
#plt.show()

# Gráfico 2: Throughput over time
plt.figure(figsize=(20, 10))
plt.plot(range(maxSteps), tp_history, marker="", linestyle="-")
plt.title('Throughput over time')
plt.xlabel('Steps')
plt.ylabel('Throughput (bits)')
plt.savefig('throughput_plot_Daniel.pdf')
#plt.show()

# Gráfico 3: RTT over time
plt.figure(figsize=(20, 10))
plt.plot(range(maxSteps), rtt_history, marker="", linestyle="-")
plt.title('RTT over time')
plt.xlabel('Steps')
plt.ylabel('RTT (microseconds)')
plt.savefig('rtt_plot_Daniel.pdf')
#plt.show()

# Gráfico 4: Reward sum plot
plt.figure(figsize=(20, 10))
plt.plot(range(maxSteps), rew_history, marker="", linestyle="-")
plt.title('Reward sum plot')
plt.xlabel('Steps')
plt.ylabel('Accumulated reward')
plt.savefig('reward_plot_Daniel.pdf')
#plt.show()

plt.figure(figsize=(20, 10))
plt.plot(range(maxSteps), mse_histori, marker="", linestyle="-")
plt.title('mse_histori')
plt.xlabel('Steps')
plt.ylabel('mse')
plt.savefig('mse_plot_Daniel.pdf')

flat_loss_history = [value for sublist in loss_history for value in sublist]

min_loss = min(flat_loss_history)
max_loss = max(flat_loss_history)
		
normalized_loss_history =[]
for sublist in loss_history:
	temp=[]
	for value in sublist:
		temp.append(maxSteps * (value - min_loss) / (max_loss - min_loss))


	normalized_loss_history.append(temp)

# Certifique-se de que normalized_loss_history tenha o mesmo número de elementos que range(maxSteps)

#print(normalized_loss_history)
#print(len(normalized_loss_history))
plt.figure(figsize=(20, 10))
vet=[]
for x in loss_history:
	for y in x:
		vet.append(y)
#print(vet,len(vet))
print(normalized_loss_history[1])
x_values = range(30)
x_labels = [str(i) if i % 2 == 0 else "" for i in x_values]  # Exibe apenas rótulos pares
plt.xticks(x_values, x_labels)
plt.plot(range(100),flat_loss_history[:100], marker="", linestyle="-")
#for i,y in range(len(normalized_loss_history)):
    #plt.plot(range(30),y, marker="", linestyle="-", label=f'Posição {i}')
plt.title('loss_histori')
plt.xlabel('Steps')
plt.ylabel('loss')

plt.savefig('loss_plot_Daniel2.pdf')

"""
"""
mpl.rcdefaults()
mpl.rcParams.update({'font.size': 16})

# Gráfico 1: Congestion windows
plt.figure(figsize=(20, 10))
plt.plot(range(len(cWnd_history)), cWnd_history, marker="", linestyle="-")
plt.title('Congestion windows')
plt.xlabel('Steps')
plt.ylabel('CWND (segments)')
plt.savefig('cwnd_plot_Daniel.pdf')
#plt.show()

# Gráfico 2: Throughput over time
plt.figure(figsize=(20, 10))
plt.plot(range(len(tp_history)), tp_history, marker="", linestyle="-")
plt.title('Throughput over time')
plt.xlabel('Steps')
plt.ylabel('Throughput (bits)')
plt.savefig('throughput_plot_Daniel.pdf')
#plt.show()

# Gráfico 3: RTT over time
plt.figure(figsize=(20, 10))
plt.plot(range(len(rtt_history)), rtt_history, marker="", linestyle="-")
plt.title('RTT over time')
plt.xlabel('Steps')
plt.ylabel('RTT (microseconds)')
plt.savefig('rtt_plot_Daniel.pdf')
#plt.show()

# Gráfico 4: Reward sum plot
plt.figure(figsize=(20, 10))
plt.plot(range(len(rew_history)), rew_history, marker="", linestyle="-")
plt.title('Reward sum plot')
plt.xlabel('Steps')
plt.ylabel('Accumulated reward')
plt.savefig('reward_plot_Daniel.pdf')
#plt.show()
plt.figure(figsize=(20, 10))
plt.plot(range(len(mse_histori)), mse_histori, marker="", linestyle="-")
plt.title('mse_histori')
plt.xlabel('Steps')
plt.ylabel('mse')
plt.savefig('mse_plot_Daniel.pdf')
#
plt.figure(figsize=(20, 10))
plt.plot(range(len(loss_history)), loss_history, marker="", linestyle="-")
plt.title('loss_histori')
plt.xlabel('Steps')
plt.ylabel('loss')
plt.savefig('loss_plot_Daniel.pdf')

"""


"""
mpl.rcdefaults()
mpl.rcParams.update({'font.size': 16})

# Gráfico 1: Congestion windows
plt.figure(figsize=(20, 10))
plt.plot(range(len(cWnd_history)), cWnd_history, marker="", linestyle="-")
plt.title('Congestion windows')
plt.xlabel('Steps')
plt.ylabel('CWND (segments)')
plt.savefig('cwnd_plot_Daniel.pdf')
#plt.show()

# Gráfico 2: Throughput over time
plt.figure(figsize=(20, 10))
plt.plot(range(len(tp_history)), tp_history, marker="", linestyle="-")
plt.title('Throughput over time')
plt.xlabel('Steps')
plt.ylabel('Throughput (bits)')
plt.savefig('throughput_plot_Daniel.pdf')
#plt.show()

# Gráfico 3: RTT over time
plt.figure(figsize=(20, 10))
plt.plot(range(len(rtt_history)), rtt_history, marker="", linestyle="-")
plt.title('RTT over time')
plt.xlabel('Steps')
plt.ylabel('RTT (microseconds)')
plt.savefig('rtt_plot_Daniel.pdf')
#plt.show()

# Gráfico 4: Reward sum plot
plt.figure(figsize=(20, 10))
plt.plot(range(len(rew_history)), rew_history, marker="", linestyle="-")
plt.title('Reward sum plot')
plt.xlabel('Steps')
plt.ylabel('Accumulated reward')
plt.savefig('reward_plot_Daniel.pdf')
#plt.show()
plt.figure(figsize=(20, 10))
plt.plot(range(len(mse_histori)), mse_histori, marker="", linestyle="-")
plt.title('mse_histori')
plt.xlabel('Steps')
plt.ylabel('mse')
plt.savefig('mse_plot_Daniel.pdf')
#
plt.figure(figsize=(20, 10))
plt.plot(range(len(loss_history)), loss_history, marker="", linestyle="-")
plt.title('loss_histori')
plt.xlabel('Steps')
plt.ylabel('loss')
plt.savefig('loss_plot_Daniel.pdf')


"""
