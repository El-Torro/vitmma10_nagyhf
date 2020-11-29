from __future__ import print_function

import glob
import os
import sys
from time import time
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

from carla import ColorConverter as cc
from controller import PidController

import random
import time
import numpy as np
import cv2

import pygame
from pygame.locals import K_ESCAPE
from threading import Lock

from lanedetector import laneDetect

IM_WIDTH = 640
IM_HEIGHT = 480
seq = 0
x_dest = 0

controller = PidController(0.001, 0.001, 0.000001)

mutex = Lock()

def process_img(image):
    if not mutex.acquire(False):
        return 0
    
    global seq
    global x_dest
    global controller
    
    white = (255, 255, 255) 
    green = (0, 255, 0) 
    blue = (0, 0, 128) 
    font = pygame.font.Font('freesansbold.ttf', 32)
    
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4)) 
    if array is None:
        return 0
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    #surface_1_1 = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    #display_surface.blit(surface_1_1, (0, 0))
    img_proc, ldArray, x_dest = laneDetect(array, x_dest) 
    if ldArray is None:
        return 0
    
    black_img = np.array(array)
    black_img[:,:] = 0
    surface_2_1 = pygame.surfarray.make_surface(black_img.swapaxes(0, 1))
    display_surface.blit(surface_2_1, (0, 480))   
    
    text = font.render(str(x_dest), True, green, blue)
    textRect = text.get_rect()
    textRect.center = (300, 530) 
    display_surface.blit(text, textRect)
    
    x_u = controller.check( x_dest )
    
    
    text = font.render(str(x_u), True, green, blue)
    textRect = text.get_rect()
    textRect.center = (300, 580) 
    display_surface.blit(text, textRect)

    vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer= x_u))
    
    surface_1_1 = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display_surface.blit(surface_1_1, (0, 0))   
    surface_1_2 = pygame.surfarray.make_surface(ldArray.swapaxes(0, 1))
    display_surface.blit(surface_1_2, (640, 0))   
    surface_2_2 = pygame.surfarray.make_surface(img_proc.swapaxes(0, 1))
    display_surface.blit(surface_2_2, (640, 480)) 
    #pygame.display.flip()
    #pygame.display.update()
     
    #print(seq)
    #seq = seq + 1
    # world.tick()  # In case of synchronous simulation
    mutex.release()
    
    return 0

def ontick(ws):
    ##print(ws.frame)
    return
    
actor_list = []
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    #client.load_world('Town04')
    
    
    world = client.get_world()
    
    # Set synchronous mode
    settings = world.get_settings()
    #settings.synchronous_mode = True
    #settings.fixed_delta_seconds = 0.05
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    world.on_tick(ontick)

    blueprint_library = world.get_blueprint_library()

    # Select Tesla model 3 from library
    bp = blueprint_library.filter('model3')[0]
    ##print(bp)

    # Select spawning point
    spawn_point = world.get_map().get_spawn_points()[3]

    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(False)  # if you just wanted some NPCs to drive.

    actor_list.append(vehicle)

    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    blueprint = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint.set_attribute('fov', '110')
    #blueprint.set_attribute('sensor_tick', '0.2')

    # Adjust sensor relative to vehicle
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
    
    # add sensor to list of actors
    actor_list.append(sensor)
    
    vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.0))
    
    # Init display
    pygame.init() 
    display_surface = pygame.display.set_mode((1280, 960), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption('CARLA image') 

    #print("Waiting for system startup")
    #time.sleep(5)
    
    # do something with this sensor
    sensor.listen(lambda data: process_img(data))
    
    # world.tick() # in case of synchronous simulation
    
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick_busy_loop(60)
        # pygame event get should be called, otherwise pygame hangs up
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            # This thing does not work here...
            elif event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    print("Game Over")
                    run = False
        # Display the picture
        pygame.display.flip()

finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')