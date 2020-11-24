from time import time


class PidController(object):
    def __init__(self, p, i, d):
        """
        A simple, pythonic PID controller
        https://en.wikipedia.org/wiki/PID_controller
        
        Constructor takes 3 arguments, i.e. the 3 pid parameters: p, i and d
        """
        self._kp = p
        self._ki = i
        self._kd = d
        self._last_value = 0
        self._last_time  = time()
    
    
    def check(self, delta):
        """
        The _check_ method takes as input parameter the delta between
        the setpoint and the current value of the process variable (i.e.
        SP-PV). It returns the corresponding value of the pid controller,
        that you could use to control a motor, a temperature, etc...
        """
        dt = time() - self._last_time
        
        return_value = self._kp * delta
        return_value += self._ki * self._integrate(delta, dt)
        return_value += self._kd * self._derive(delta, dt)
        
        self._last_value = delta
        self._last_time  = time()
        
        return return_value
    
    
    def _integrate(self, new_value, dt):
        return 0.5*dt*(self._last_value+new_value)

    def _derive(self, new_value, dt):
        return (new_value-self._last_value)/dt