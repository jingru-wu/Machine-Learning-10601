"""
Example command：
<mode> <weight out> <returns out> <episodes> <max iterations> <epsilon> <gamma> <learning rate>.
$ python q_learning.py raw weight.out returns.out 4 200 0.05 0.99 0.01
1. <mode>: mode to run the environment in. Should be either ‘‘raw’’ or ‘‘tile’’.
2. <weight out>: path to output the weights of the linear model.
3. <returns out>: path to output the returns of the agent
4. <episodes>: the number of episodes your program should train the agent for.
5. <max iterations>: the maximum of the length of an episode. When this is reached, we terminate the current episode.
6. <epsilon>: the value  for the epsilon-greedy strategy
7. <gamma>: the discount factor γ.
8. <learning rate>: the learning rate α of the Q-learning algorithm
"""
import sys, random
import matplotlib.pyplot as plt
from environment import *
def main(argv):
	model_select = argv[1]
	weight_out = open(argv[2],'w')
	return_out = open(argv[3],'w')
	num_epsodides = int(argv[4])
	max_interation = int(argv[5])
	epsilon = float(argv[6])
	gama = float(argv[7])  # discount
	l_rate = float(argv[8])

	env=MountainCar(model_select)
	env.__init__(model_select)
	num_state=env.state_space
	num_action=env.action_space
	weight,bias=initialize(num_state,num_action)
	reward_list=[]
	for i in range(num_epsodides):
		cur_state=env.reset()
		cur_state=dict2array(cur_state,num_state)
		count=0
		flag=False
		total_reward=0
		while flag!=True:
			Q_current=q(cur_state,weight,bias)
			action=next_action(Q_current,epsilon)
			(next_state,reward,flag)=env.step(action)
			total_reward+=reward
			next_state=dict2array(next_state,num_state)
			Q_next=q(next_state,weight,bias)
			A=l_rate*(Q_current[action]-(reward+gama*max(Q_next)))
			weight[:,action]-= A*(cur_state.flatten())
			bias-=A
			cur_state=next_state
			count+=1
			if count>=max_interation:
				flag=True
		# print(count)
		return_out.write(str(total_reward)+'\n')
		reward_list.append(total_reward)
	r=np.linspace(0,num_epsodides-1,num_epsodides)
	r.astype(int)
	mean_array=running_mean(reward_list,25)
	reward_plot, = plt.plot(r,reward_list,label='Reward')
	mean_plot, = plt.plot(r,list(mean_array),'r',label='Moving Mean Reward')
	plt.legend([reward_plot, mean_plot], ['Reward', 'Moving Mean Reward'])
	plt.ylabel('Return Reward')
	plt.xlabel('Episode')
	plt.title('Reward VS Episode in Tile model')
	plt.show()
	#
	# print('bias:',bias)
	# string=''+str(bias)+'\n'
	# weight_list=list(weight.flatten())
	# for element in weight_list:
	# 	print(element)
	# 	string+=str(element)+'\n'
	# weight_out.write(string)
	# weight_out.close()
	# return_out.close()

def next_action(Q, epsilon):
	if random.random() < epsilon:
		return random.randint(0, 2)
	else:
		return np.argmax(Q)

def initialize(state_space,action_space):
	weight=np.zeros((state_space,action_space))
	bias=0
	return weight, bias

def q(s,weight,bias):
	"""
	:input:  S:1*S
			Weight: S*A
			bias: singular
			B: A*1
	:return: Q: A*1
	"""
	B=bias*np.ones((1,weight.shape[1]))
	res=np.dot(s,weight)+B # 1*S  S*A + 1*A
	return res.flatten()

def running_mean(x, N):
	res=np.ones(len(x))*(-200)
	x=np.asarray(x)
	cumsum = np.cumsum(np.insert(x, 0, 0))
	res[24:]=(cumsum[N:] - cumsum[:-N]) / float(N)
	return res

def dict2array(dict,num_state):
	array=np.zeros((1,num_state))
	for key in dict.keys():
		array[0,key]=dict[key]
	return array

if __name__ == "__main__":
	# main(sys.argv)
	argv=['','tile','raw_weight.out','returns.txt','400','200','0.05','0.99','0.00005']
	main(argv)
