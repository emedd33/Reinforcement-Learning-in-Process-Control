from main_imports import *
from params import * # Parameters used in main

def main():
    #============= Initialize variables ===========#
    tank = Tank(TANK_HEIGHT,TANK_RADIUS) # get model
    simulator = Simulator()
     
     # initialize prediction model
    model = ANN_model(
        input_size=OBSERVATIONS-1, #uses the gradient n observations
        hidden_size=[10,10],
        output_size=VALVE_POSITIONS,
        max_level=tank.h
        )
    
    # ================= Running episodes =================#
    running=True
    rewards = [] 
    for e in range(EPISODES):
        simulator.reset() # Reset level in tank
        
        level_history = OBSERVATIONS*[tank.l] 
        valve_history = OBSERVATIONS*[0.5]
        # Running through states in the episode
        ad_counter = simulator.action_delay
        for t in range(MAX_TIME):
            state = level_history[-OBSERVATIONS:] # Observe the last states
            if ad_counter >= simulator.action_delay:
                action = model.predict(state) 
                simulator.action_delay_counter = -1
            # simulator.remember(action,state)
            next_state = simulator.get_next_state(action) # play out the action
            
            # Check terminate state
            if next_state == "Terminated":
                break

            if simulator.show_rendering:
                simulator.render(action,next_state)

            valve_history.append(action)
            level_history.append(next_state)   
        rewards.append(t)

        # Live plot rewards
        if simulator.live_plot:
            simulator.plot(t,e)
        if not simulator.running:
            break
                
        
        
    pygame.display.quit()
    print("\nMean rewards for episodes: ", np.mean(rewards)) 
    print("Rewards for the last episode: ", rewards[-1])

if __name__ == "__main__":
    main()