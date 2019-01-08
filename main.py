from models.environment import Environment
from models.Agent import Agent
from params import * # Parameters used in main
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    #============= Initialize variables ===========#
    
    environment = Environment()
    agent = Agent()
    # ================= Running episodes =================#
    running=True
    rewards = [] 
    for e in range(EPISODES):
        environment.reset() # Reset level in tank
        
        level_history = OBSERVATIONS*[environment.model.l] 
        valve_history = OBSERVATIONS*[0.5]
        # Running through states in the episode
        for t in range(MAX_TIME):
            state = level_history[-OBSERVATIONS:] # Observe the last states
            if environment.action_delay_counter >= environment.action_delay:
                action = agent.predict(state) 
                environment.action_delay_counter = -1

                # Save chosen action with state
                agent.state_memory.append(state)
                agent.action_memory.append(action)
            next_state = environment.get_next_state(action) # play out the action
            
            # Check terminate state
            if next_state == "Terminated":
                break

            if environment.show_rendering:
                environment.render(action,next_state)

            valve_history.append(action)
            level_history.append(next_state)   
        rewards.append(t)

        # Live plot rewards
        if environment.live_plot:
            environment.plot(t,e)
        if not environment.running:
            break
                
        
        
    pygame.display.quit()
    print("\nMean rewards for episodes: ", np.mean(rewards)) 
    print("Rewards for the last episode: ", rewards[-1])

if __name__ == "__main__":
    main()