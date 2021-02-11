# Module import
import gym
import numpy as np

# Environment
env = gym.make('FrozenLake-v0')

print(env.observation_space.n)  # 4 x 4 Environment(16)
print(env.action_space.n) # left, up, right, down (4)



def compute_value_fucntion(env,gamma = 1.0):
    # Value iteration
    # Initialize Value table
    value_table = np.zeros(env.obervation_space.n)
    no_of_iterations = 100000 # iteration times
    threshold = 1e-20

    # 각 iteration시작할 때 value table을 updated_value_table로 복사함
    for i in range(no_of_iterations):
        updated_value_table = np.copy(value_table)

    # 어떤 State s에서 모든 행동들에 대한 next_states_rewards를 계산하여 Q값에 추가 할 때 최대 Q값을 선택하여 그 상태의 가치로 저장합니다.
    for state in range(env.observation_space.n):
        Q_value = []
        for action in range(env.action_space.n):
            next_states_rewards = []
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                next_states_reward.append((trans_prob * (reward_prob + gamma * updated_value_table[next_state])))
                Q_value.append(np.sum(next_states_rewards))
                # 최대가 되는 Q값을 선택
                value_table[state] = max(Q_value)
            if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
                print('Value-iteration converged at interation# %d.' %(i+1))
                break
        return value_table, Q_value

    def extract_policy(value_table, gamma = 1.0):
        policy = np.zeros(env.observation_space.n)
        for state in range(env.observation_space.n):
            Q_table = np.zeros(env.action_space.n)
            for action in range(env.action_space.n):
                for next_sr in env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))
            policy[state] = np.argmax(Q_table)
        return policy
    optimal_value_function = value_iteration(env = env, gamma = 1.0)
    optimal_policy = extract_policy(optimal_value_function, gamma = 1.0)
    print(optimal_policy)

    def policy_iteration(env, gamma = 1.0):
        random_policy = np.zeros(env.observation_space.n)
        no_of_iterations = 200000
        gamma = 1.0
        for i in range(no_of_iterations):
            new_value_function = compute_value_fucntion(random_policy, gamma)
            new_policy = extract_policy(new_value_function, gamma)
            if (np.all(random_policy == new_policy)):
                print('Policy-Iteration converged at step %d.' %(i+1))
                break
            random_policy = new_policy
        return new_policy
print(policy_iteration(env))
