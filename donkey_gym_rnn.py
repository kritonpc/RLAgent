import gym, gym_donkeycar, donkeycar_modified
from agent import DQNAgent
import tensorflow as tf
import cv2, random, time
import numpy as np
from csv import writer;
from utils import *;

def calculateThrottle(velocity, max_velocity, max_acceleration):
	return min((max_velocity - velocity)/12.5 + 0.05, max_acceleration) * (velocity < max_velocity)

def main():
	# enable GPU memory growth
	physical_devices = tf.config.list_physical_devices('GPU') 
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
	
	# model & training information
	model_name = input("Model name -> ");
	load_trained = input("Load trained (y/n)? ").lower() == "y";
	epsilon = float(input("Epsilon -> "));
	episode_count = int(input("Episode count -> "));
	
	model_location = "models/" + model_name + "/";
	model_path = model_location + ("model_trained.h5" if load_trained else "model.h5");
	
	print("Loading", model_path, "with epsilon", epsilon);
	agent = DQNAgent(model_path, epsilon);
	
	try: agent.memory = json.load(model_location + "data.json");
	except: agent.memory = [];
	
	# training information
	resizeScale = (40, 30);
	batch_size = 12;
	frame_n = 3;
	max_cte = 4.35;
	# max_cte = 3.5;
	
	# statistics
	score = [];
	rewards = [];
	highest_score = 0;
	highest_reward = 0;
	max_score = None;
	
	# velocity
	max_velocity = 10.0
	max_acceleration = 0.75
	
	# steering
	max_steering = 0.75
	steering_step = 2*max_steering/(agent.action_space-1)
	steering_table = [i*steering_step-max_steering for i in range(agent.action_space)]
	
	file = open("log.csv", "w+", newline="");
	log = writer(file);
	log.writerow(['Episode','Timestep', 'Avg Steer', 'Min Reward', 'Avg Reward', 'Max Reward', 'Episode Length', 'Reward Sum', 'Max Q steer', 'Max Q throttle', 'Epsilon','Episode Time', 'Avg Speed','Max Speed','Min CTE','Avg CTE','Max CTE','Distance', "Average Throttle", "Max Throttle", "Min Throttle", "Average Absolute CTE", "Min Absolute CTE", "Max Absolute CTE"]);
	
	# setup donkey environment
	conf = {
		# "exe_path":"remote",
		"exe_path":"D:/sdsandbox/build2/donkey_sim.exe",
		"host":"127.0.0.1",
		"port":9091,
		"body_style":"donkey",
		"body_rgb":(128, 128, 128),
		"car_name":"rl",
		"font_size":100
	}

	# env = gym.make("donkey-generated-roads-v0", conf=conf)
	env = gym.make("donkey-generated-track-v0", conf=conf)
	env.viewer.handler.max_cte = max_cte;
	cv2.namedWindow("camera");
	
	first_train = True;
	first_start = time.time();
	timestep = 0;
	success_episodes = 0;
	max_laps = 5;
	
	for e in range(episode_count):
		# at each episode, reset the environment
		state = env.reset();
		states = np.empty((frame_n, resizeScale[1], resizeScale[0], 3));
		states[0] = preprocessImage(state, resizeScale);
		need_frames = frame_n-1;

		done = False;
		score.append(0);
		rewards.append(0.0);
		last_velocity = [0.0];
		laps = 0;
		start = time.time();
		
		# logging
		steers = [];
		throttles = [];
		rewards_ = [];
		velocities = [];
		ctes = [];
		ctes_absolute = [];
		max_q_steer = 0.0;
		distance = 0.0;
		distance_time = start;
		
		while not done and (score[-1] < max_score if max_score else True):
			if need_frames > 0:
				next_state, reward, done, info = env.step([steering_table[random.randint(0, agent.action_space-1)], 0.15]);
				
				states[frame_n-need_frames] = preprocessImage(next_state, resizeScale);
				need_frames -= 1
				
				last_velocity.append(info["speed"]);
				continue
		
			# select action, observe environment, calculate reward
			action, Q = agent.act(np.asarray([states]));
			steering = steering_table[action];
			throttle = calculateThrottle(last_velocity[-1], max_velocity, max_acceleration);
			
			next_state, reward, done, info = env.step([steering, throttle]);
			last_velocity.append(round(info["speed"], 4));
			
			img = cv2.resize(next_state, (320, 240), interpolation=cv2.INTER_AREA)
			cv2.imshow("camera", img)
			
			reward = 0.0 if not done else -1.0;
			if abs(info["cte"]) >= max_cte:
				done = True
				reward = -1.0
			
			if not done:
				reward = (1.0 - (abs(info["cte"]) / max_cte));
			
			# for track
			if info["lap_finished"]:
				laps += 1;
				if laps == max_laps:
					done = True
			
			timestep += 1;
			score[-1] += 1;
			rewards[-1] += reward;
			
			next_states = np.roll(states, -1, axis=0);
			next_states[-1] = preprocessImage(next_state, resizeScale);
			
			# save experience and update current state
			agent.remember([states], action, reward, [next_states], done);
			states = next_states;
			
			if not first_train:
				agent.replay(batch_size);
			
			# logging
			steers.append(steering);
			throttles.append(throttle);
			rewards_.append(reward);
			velocities.append(last_velocity[-1]);
			ctes.append(info["cte"]);
			ctes_absolute.append(abs(info["cte"]));
			distance += last_velocity[-1]*(time.time()-distance_time)
			distance_time = time.time()
			
			if Q != None and (max_q_steer == None or Q > max_q_steer):
				max_q_steer = Q;
			
			cv2.waitKey(1);
			
		# for roads
		# if distance > 1900:
			# laps = max_laps
			
		# logging
		if score[-1] > 0:
			log.writerow([e, timestep, round(mean(steers), 2), round(min(rewards_), 2), round(mean(rewards_), 2), round(max(rewards_), 2), score[-1], round(rewards[-1], 2), round(max_q_steer, 2), 0, agent.epsilon, round(time.time()-start,2), round(mean(velocities), 2), round(max(velocities), 2), round(min(ctes), 2), round(mean(ctes), 2), round(max(ctes), 2), round(distance, 2), round(mean(throttles), 2), round(max(throttles), 2), round(min(throttles), 2), round(mean(ctes_absolute), 2), round(min(ctes_absolute), 2), round(max(ctes_absolute), 2)]);
		else: # sometimes, something goes really wrong... don't count this episode
			e -= 1
		
		file.flush();
		
		# fix for persisting throttle bug
		env.step([0.0, -0.03]);
		
		if len(agent.memory) > batch_size*4 and first_train:
			agent.replay(batch_size);
			agent.act(np.asarray([states]));
			first_train = False;
		
		if len(score) > 20: score = score[-20:];
		if len(rewards) > 20: rewards = rewards[-20:];
			
		if score[-1] >= highest_score:
			highest_score = score[-1];
		
		if rewards[-1] >= highest_reward:
			highest_reward = rewards[-1]
			agent.save();
		
		print("episode: {}/{}, steps: {}, reward: {}, highest reward: {}, average: {}, laps: {}, e: {:.2}, memory: {}, replays: {}"
				.format(e+1, episode_count, score[-1], round(rewards[-1], 2), round(highest_reward, 2), round(mean(rewards), 2), laps, agent.epsilon, len(agent.memory), agent.replays));
		
		if (e+1) % 5 == 0:
			print("Took", round((time.time()-start)/60, 2), "minutes\n");
			start = time.time();
			agent.merge_models();
		
		if laps == max_laps:
			success_episodes += 1;
		else:
			success_episodes = 0;
		
		if success_episodes == 5:
			print("Training successfull! Time: {} minutes.".format(round((time.time()-first_start)/60.0, 2)));
			agent.save("end.h5");
			file.close();
			break;

	agent.save();
	print("Total training time:", round((time.time()-first_start)/60, 2), "minutes");

if __name__ == "__main__":
	main()