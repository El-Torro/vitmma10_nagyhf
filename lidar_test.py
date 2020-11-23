import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
import time
import numpy as np
import cv2

import pygame
from pygame.locals import K_ESCAPE
from threading import Lock

import open3d


IM_WIDTH = 640
IM_HEIGHT = 480

mutex = Lock()

def process_lidar(image):
    if not mutex.acquire(False):
        return 0
    
    #data.save_to_disk("cloud.ply")
    points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 3), 3))
    lidar_data = np.array(points[:, :2])
    lidar_data *= min((640,480)) / (2.0 * 20) # Lidar range is 20 m
    lidar_data += (0.5 * 640, 0.5 * 480)
    lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
    lidar_data = lidar_data.astype(np.int32)
    lidar_data = np.reshape(lidar_data, (-1, 2))
    lidar_img_size = (640, 480, 3)
    lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
    lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
    
    surface = pygame.surfarray.make_surface(lidar_img)    
    display_surface.blit(surface, (0, 0))    

    mutex.release()
    
    return 0

def process_img(image):
    if not mutex.acquire(False):
        return 0
        
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display_surface.blit(surface, (640, 0))    
     
    mutex.release()
    
    return 0
    
def ontick(ws):
    print(ws.frame)
    
actor_list = []
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)

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
    print(bp)

    # Select spawning point
    spawn_point = world.get_map().get_spawn_points()[0]

    vehicle = world.spawn_actor(bp, spawn_point)
    #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.

    actor_list.append(vehicle)

    # get the blueprint for this sensor
    blueprint = blueprint_library.find('sensor.lidar.ray_cast')
    blueprint.set_attribute('channels',str(32))
    blueprint.set_attribute('points_per_second',str(90000))
    blueprint.set_attribute('rotation_frequency',str(40))
    blueprint.set_attribute('range',str(20))
    # Adjust sensor relative to vehicle
    spawn_point = carla.Transform(carla.Location(z=2.0))

    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
    
    # add sensor to list of actors
    actor_list.append(sensor)


    blueprint = blueprint_library.find('sensor.camera.rgb')
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint.set_attribute('fov', '110')
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
    sensor_cam = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
    actor_list.append(sensor_cam)
        
    # Init display
    pygame.init() 
    display_surface = pygame.display.set_mode((1280, 480), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption('CARLA image') 

    # do something with this sensor
    sensor.listen(lambda data: process_lidar(data))
    sensor_cam.listen(lambda data: process_img(data))
    
    # world.tick() # in case of synchronous simulation

    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick_busy_loop(60)
        # pygame event get should be called, otherwise pygame hangs up
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
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