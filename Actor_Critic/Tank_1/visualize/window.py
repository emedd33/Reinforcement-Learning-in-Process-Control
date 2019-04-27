import pygame


class Window:
    def __init__(self, tank):
        pygame.init()
        pygame.display.set_caption("Tank simulation")
        self.WINDOW_HEIGHT = 400
        self.WINDOW_WIDTH = 300
        self.screen = pygame.display.set_mode(
            (self.WINDOW_HEIGHT, self.WINDOW_WIDTH)
        )
        self.background_image = pygame.image.load(
            "Actor_Critic/Tank_1/visualize/images/EmptyTank.png"
        ).convert()
        self.background_image = pygame.transform.scale(
            self.background_image, (self.WINDOW_HEIGHT, self.WINDOW_WIDTH)
        )
        self.clock = pygame.time.Clock()
        self.tank = TankImage(tank, 56.5, 29)

    def Draw(self, input_z):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        self.screen.blit(self.background_image, [0, 0])
        self.tank.draw(self.screen, input_z)

        pygame.display.flip()
        return True


class TankImage:
    height = 200
    width = 148
    choke_width = 35
    choke_height = 5
    rga_water = (25, 130, 150)
    rga_choke = (0, 0, 0)
    choke_left_adj = 234
    choke_top_adj = -7
    choke_range = 167

    def __init__(self, tank, left_pos, top_pos):
        self.tank = tank[0]
        self.left_pos = left_pos
        self.top_pos = top_pos

    def draw(self, screen, z):
        self.draw_level(screen)
        self.draw_choke(screen, z[0])

    def draw_level(self, screen):
        level_percent = (self.tank.level - self.tank.min) / (
            self.tank.max - self.tank.min
        )
        pygame.draw.rect(
            screen,
            TankImage.rga_water,
            pygame.Rect(
                self.left_pos,
                self.top_pos + (1 - level_percent) * TankImage.height,
                TankImage.width,
                level_percent * TankImage.height,
            ),
        )

    def draw_choke(self, screen, z):
        pygame.draw.rect(
            screen,
            TankImage.rga_choke,
            pygame.Rect(
                self.left_pos + TankImage.choke_left_adj,
                self.top_pos
                + TankImage.choke_top_adj
                + (1 - z) * TankImage.choke_range,
                TankImage.choke_width,
                TankImage.choke_height,
            ),
        )
