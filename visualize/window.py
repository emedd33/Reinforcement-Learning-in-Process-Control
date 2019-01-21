import pygame
class Window():
    def __init__(self,tank):
        pygame.init()
        pygame.display.set_caption('Tank simulation')
        self.WINDOW_HEIGHT=WINDOW_HEIGHT=400
        self.WINDOW_WIDTH=WINDOW_WIDTH=300
        self.screen = pygame.display.set_mode((self.WINDOW_HEIGHT, self.WINDOW_WIDTH))
        self.background_image = pygame.image.load("visualize/images/EmptyTank.png").convert()
        self.background_image = pygame.transform.scale(self.background_image, (WINDOW_HEIGHT, WINDOW_WIDTH))
        
        # The parameters set to this are hardcoded and should not be changed
        self.TANK_LEFT_POS = WINDOW_WIDTH/6.3
        self.TANK_TOP_POS = WINDOW_HEIGHT/14
        self.TANK_HEIGHT=200
        self.TANK_WIDTH=157
        self.TANK_BOARDER = 10

        self.MEASER_BOARDER = 10
        self.MEASER_HEIGHT=150

        self.CHOKE_CLOSED_POS = self.TANK_TOP_POS + self.TANK_HEIGHT*0.8
        self.CHOKE_WIDTH = 35
        self.CHOKE_HEIGHT=5
        self.CHOKE_LEFT_POS = WINDOW_WIDTH*0.97
        self.CHOKE_RANGE = 165

        self.RGA_CHOKE=(0, 0, 0)
        self.RGA_WATER = (25,130,150)
        
        self.clock  = pygame.time.Clock()
        self.tank = tank

    def Draw(self,input_z,level):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                    return False
        self.screen.blit(self.background_image, [0, 0])
        # self.DrawTank()
        self.DrawLevel(level)
        self.DrawChoke(input_z)
        pygame.display.flip()
        return True

    def DrawChoke(self,input_z):
        delta_z = self.CHOKE_RANGE*input_z
        choke_pos = self.CHOKE_CLOSED_POS -delta_z
        
        pygame.draw.rect(self.screen,self.RGA_CHOKE,
        pygame.Rect(
            self.CHOKE_LEFT_POS,
            choke_pos,
            self.CHOKE_WIDTH,
            self.CHOKE_HEIGHT
            ))
    def DrawLevel(self,level): #TODO change max min level visual 
        level_percent = level/self.tank.h
        draw_level = int(level_percent*self.TANK_HEIGHT)
        pygame.draw.rect(self.screen,self.RGA_WATER,
        pygame.Rect(
            self.TANK_LEFT_POS+self.TANK_BOARDER,
            self.TANK_TOP_POS+self.TANK_HEIGHT-draw_level+1,
            self.TANK_WIDTH-self.TANK_BOARDER,
            draw_level
            ))
    def DrawTank(self):
        
        pygame.draw.rect(self.screen,self.RGA_TANK,
        pygame.Rect(
            self.TANK_LEFT_POS,
            self.TANK_TOP_POS,
            self.TANK_BOARDER,
            self.TANK_HEIGHT))
        pygame.draw.rect(self.screen,self.RGA_TANK,
        pygame.Rect(
            self.TANK_LEFT_POS+self.TANK_WIDTH,
            self.TANK_TOP_POS,
            self.TANK_BOARDER,
            self.TANK_HEIGHT))
        pygame.draw.rect(self.screen,self.RGA_TANK,
        pygame.Rect(
            self.TANK_LEFT_POS,
            self.TANK_TOP_POS+self.TANK_HEIGHT,
            self.TANK_WIDTH+self.TANK_BOARDER,
            self.TANK_BOARDER
        ))
