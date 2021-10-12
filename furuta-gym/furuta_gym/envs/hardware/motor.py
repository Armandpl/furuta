import RPi.GPIO as GPIO


class Motor():
    def __init__(self, D2, IN1, IN2, freq=500):
        self.D2 = D2
        self.IN1 = IN1
        self.IN2 = IN2

        GPIO.setmode(GPIO.BCM)

        GPIO.setup(self.IN1, GPIO.OUT)
        GPIO.setup(self.IN2, GPIO.OUT)
        GPIO.setup(self.D2, GPIO.OUT)

        self.pwm = GPIO.PWM(self.D2, freq)
        self.pwm.start(0)

    def set_speed(self, speed):
        if speed >= 0:
            self.set_direction(1)
        elif speed < 0:
            self.set_direction(-1)

        speed = abs(speed)*100.0
        self.pwm.ChangeDutyCycle(speed)

    def set_direction(self, direction):
        if direction == -1:
            GPIO.output(self.IN2, GPIO.LOW)
            GPIO.output(self.IN1, GPIO.HIGH)
        elif direction == 1:
            GPIO.output(self.IN1, GPIO.LOW)
            GPIO.output(self.IN2, GPIO.HIGH)

    def close(self):
        self.set_speed(0)
        GPIO.cleanup()
