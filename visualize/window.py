import pygame
class Window():
    def __init__(self,tank,WINDOW_HEIGHT,WINDOW_WIDTH):
        pygame.init()
        self.WINDOW_HEIGHT=WINDOW_HEIGHT
        self.WINDOW_WIDTH=WINDOW_WIDTH
        self.TANK_LEFT_POS = self.WINDOW_WIDTH/2
        self.TANK_TOP_POS = self.WINDOW_HEIGHT/4
        self.TANK_HEIGHT=150
        self.TANK_WIDTH=100
        self.TANK_BOARDER = 10

        self.MEASER_BOARDER = 10
        self.MEASER_HEIGHT=150

        self.CHOKE_CLOSED = self.TANK_TOP_POS+self.MEASER_HEIGHT
        self.CHOKE_WIDTH = self.TANK_WIDTH/2
        self.CHOKE_HEIGHT=self.TANK_BOARDER
        self.CHOKE_LEFT_POS = self.TANK_LEFT_POS+self.TANK_WIDTH*1.5
        self.CHOKE_OPEN = self.TANK_TOP_POS

        self.RGA_CHOKE=(155, 88, 0)
        self.RGA_MEASER = (0,0,0)
        self.RGA_TANK = (0,0,0)
        self.RGA_WATER = (25,130,150)
        
        self.screen = pygame.display.set_mode((self.WINDOW_HEIGHT, self.WINDOW_WIDTH))
        self.clock  = pygame.time.Clock()
        self.tank = tank

    def Draw(self,input_z):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                    return False
        self.screen.fill((255,255,255))
        self.DrawTank()
        self.DrawLevel()
        self.DrawChoke(input_z)
        pygame.display.flip()
        return True

    def DrawChoke(self,input_z):
        delta_choke = self.MEASER_HEIGHT*input_z
        choke_pos = self.CHOKE_CLOSED-delta_choke
        pygame.draw.rect(self.screen,self.RGA_CHOKE,
        pygame.Rect(
            self.CHOKE_LEFT_POS+(self.CHOKE_WIDTH-self.MEASER_BOARDER)/2,
            self.CHOKE_OPEN,
            self.MEASER_BOARDER,
            self.MEASER_HEIGHT
            ))
        pygame.draw.rect(self.screen,self.RGA_MEASER,
        pygame.Rect(
            self.CHOKE_LEFT_POS,
            choke_pos,
            self.CHOKE_WIDTH,
            self.CHOKE_HEIGHT
            ))
    def DrawLevel(self):
        level_percent = self.tank.l/self.tank.h
        level = int(level_percent*self.TANK_HEIGHT)
        pygame.draw.rect(self.screen,self.RGA_WATER,
        pygame.Rect(
            self.TANK_LEFT_POS+self.TANK_BOARDER,
            self.TANK_TOP_POS+self.TANK_HEIGHT-level+1,
            self.TANK_WIDTH-self.TANK_BOARDER,
            level
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
