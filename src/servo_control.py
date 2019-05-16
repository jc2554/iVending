"""
Servo control functions

MIT License

Copyright (c) 2019 JinJie Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


import pigpio

# Constant GPIO pin for the Servo
SERVO_PIN = 12

# Connect pigpio to local Pi.
hw_pi = pigpio.pi()


# helper function to change freq and dc based on give high_pulse_width
def update_pwm(hpw):
    global hw_pi
    freq = 1/(0.020+hpw)
    dc = 1000000*hpw/(0.020+hpw)
    print("high_pulse_width="+str(hpw)+"\tfreq="+str(int(freq))+"\tdc="+str(int(dc))) # print current state
    hw_pi.hardware_PWM(SERVO_PIN, int(freq), int(dc)) # 46.5116Hz 0% dutycycle
    

# helper function to turn the camera to face downward
def turn_camera_down(ti=20000):
    global hw_pi
    print("...face_camera_down for ", ti)
    update_pwm(0.0013)
    t1 = hw_pi.get_current_tick()
    while hw_pi.get_current_tick()-t1<ti:
        pass
    update_pwm(0)


# helper function to turn the camera to face downward
def turn_camera_up(ti=20000):
    global hw_pi
    print("...face_camera_up for:", ti)
    update_pwm(0.0017)
    t1 = hw_pi.get_current_tick()
    while hw_pi.get_current_tick()-t1<ti:
        pass    
    update_pwm(0)