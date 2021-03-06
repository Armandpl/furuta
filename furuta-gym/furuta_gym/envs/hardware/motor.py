import Jetson.GPIO as GPIO


class Motor():
    def __init__(self, D2, IN1, IN2, freq=500):
        self.D2 = D2
        self.IN1 = IN1
        self.IN2 = IN2

        GPIO.setmode(GPIO.BOARD)

        GPIO.setup(self.IN1, GPIO.OUT)
        GPIO.setup(self.IN2, GPIO.OUT)
        GPIO.setup(self.D2, GPIO.OUT, initial=GPIO.HIGH)

        self.pwm = GPIO.PWM(self.D2, freq)
        self.pwm.start(0)

    def set_speed(self, speed):
        if speed >= 0:
            self.set_direction(1)
        elif speed < 0:
            self.set_direction(-1)

        # make sure the motor gets the minimum voltage
        # TODO: should depend on motor specs
        speed = abs(speed)
        if speed > 0.0:
            speed = 0.2 + 0.8 * speed

        speed = speed*100.0
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
        self.pwm.stop()
        GPIO.cleanup()


if __name__ == "__main__":
    from time import sleep

    motor = Motor(32, 29, 31)

    print("go")
    for i in range(10):
        motor.set_speed(0.3)
        sleep(1/3)
        motor.set_speed(-0.3)
        sleep(1/3)

    motor.close()
