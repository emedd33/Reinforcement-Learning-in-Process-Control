
from Tank_params import TANK1_PARAMS,TANK1_DIST,TANK2_DIST,TANK2_PARAMS,\
    TANK3_PARAMS,TANK3_DIST,TANK4_PARAMS,TANK4_DIST,TANK5_PARAMS,TANK5_DIST,TANK6_PARAMS,TANK6_DIST
from models.tank_model.tank import Tank
def create():  
    model1 = Tank(
        height=TANK1_PARAMS['height'],
        radius=TANK1_PARAMS['width'],
        max_level=TANK1_PARAMS['max_level'],
        min_level=TANK1_PARAMS['min_level'],
        pipe_radius=TANK1_PARAMS['pipe_radius'],
        dist = TANK1_DIST
        ) 
    model2 = Tank(
        height=TANK2_PARAMS['height'],
        radius=TANK2_PARAMS['width'],
        max_level=TANK2_PARAMS['max_level'],
        min_level=TANK2_PARAMS['min_level'],
        pipe_radius=TANK2_PARAMS['pipe_radius'],
        dist = TANK2_DIST,
        prev_tank=model1
        ) 
    
    model3 = Tank(
        height=TANK3_PARAMS['height'],
        radius=TANK3_PARAMS['width'],
        max_level=TANK3_PARAMS['max_level'],
        min_level=TANK3_PARAMS['min_level'],
        pipe_radius=TANK3_PARAMS['pipe_radius'],
        dist = TANK3_DIST,
        prev_tank=model2
        ) 
    model4 = Tank(
        height=TANK4_PARAMS['height'],
        radius=TANK4_PARAMS['width'],
        max_level=TANK4_PARAMS['max_level'],
        min_level=TANK3_PARAMS['min_level'],
        pipe_radius=TANK4_PARAMS['pipe_radius'],
        dist = TANK4_DIST,
        prev_tank=model3
        ) 
        
    model5 = Tank(
        height=TANK5_PARAMS['height'],
        radius=TANK5_PARAMS['width'],
        max_level=TANK5_PARAMS['max_level'],
        min_level=TANK5_PARAMS['min_level'],
        pipe_radius=TANK5_PARAMS['pipe_radius'],
        dist = TANK5_DIST,
        prev_tank=model4
        ) 
    model6 = Tank(
        height=TANK6_PARAMS['height'],
        radius=TANK6_PARAMS['width'],
        max_level=TANK6_PARAMS['max_level'],
        min_level=TANK6_PARAMS['min_level'],
        pipe_radius=TANK6_PARAMS['pipe_radius'],
        dist = TANK6_DIST,
        prev_tank=model5
        ) 
    model = []
    model.append(model1)
    model.append(model2)
    model.append(model3)
    model.append(model4)
    model.append(model5)
    model.append(model6)
    return model
    