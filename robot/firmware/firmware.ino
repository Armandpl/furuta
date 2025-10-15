#include <stdint.h>
#include <pico/time.h>
#include <Adafruit_AS5600.h>
#include <Adafruit_NeoPixel.h>

// protocol def
const uint8_t PACKET_SIZE = 6;
const uint8_t RESET = 0;
const uint8_t STEP = 1;

// Pin definitions
// motor driver
const int STEP_PIN = D0;
const int DIR_PIN = D1;
static struct repeating_timer timer; // tick timer
const int MAX_DELAY_US = 100;  // toggle every 1ms = 1 rising every 2 ms = 500 steps/s = ~1RPS = slowest
const int MIN_DELAY_US = 15;
const int MOTOR_ENC = A2;  // read with analogWrite until we get AS5600 that has configurable address
const int MOTOR_CPR = 1024;
int MOTOR_OFFSET = 0;

int MOTOR_LIMIT = MOTOR_CPR / 4;  // 180 deg of range
int MOTOR_HYS = 64; // limit hysteresis TODO rad/deg 

const unsigned long COMMAND_TIMEOUT = 500;  // ms

// Adafruit_AS5600 as5600;

// tmp LED debug
int Power = 11;
int PIN  = 12;
#define NUMPIXELS 1
Adafruit_NeoPixel pixels(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);


int pModulo(int value, int modulus) {
  return ((value % modulus) + modulus) % modulus;
}


bool timer_callback(struct repeating_timer *t) {
  digitalWrite(STEP_PIN, !digitalRead(STEP_PIN));
  return true;  // keep repeating
}


void setMotorDirection(bool direction) 
{
  if (direction == true) {digitalWrite(DIR_PIN, LOW);}
  else if (direction == false) {digitalWrite(DIR_PIN, HIGH);}
}


void processMotorCommand(float motor_command, bool direction) {
  setMotorDirection(direction);

  if (motor_command == 0) {
    cancel_repeating_timer(&timer);  // stop motor
    return;
  }

  // map absolute speed (0-1) to timer interval
  float abs_speed = motor_command / 65535;
  int timer_interval = (int)(MAX_DELAY_US - (abs_speed * (MAX_DELAY_US - MIN_DELAY_US)));

  cancel_repeating_timer(&timer);
  add_repeating_timer_us(timer_interval, timer_callback, NULL, &timer);
}

void setLED(String color) {
  pixels.clear();
  pixels.setPixelColor(0, pixels.Color(0, 0, 0));
  if (color == "red") {
    pixels.setPixelColor(0, pixels.Color(255, 0, 0));
  }
  else if (color == "green") {
    pixels.setPixelColor(0, pixels.Color(0, 255, 0));
  }
  else if (color == "blue") {
    pixels.setPixelColor(0, pixels.Color(0, 0, 255));
  }
  pixels.show();
}


void setup() {
  // setup motor pins TODO is this even needed on rp2040?
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(MOTOR_ENC, INPUT);

  pixels.begin();
  pinMode(Power,OUTPUT);
  digitalWrite(Power, HIGH);

  MOTOR_OFFSET = analogRead(MOTOR_ENC);

  //if (!as5600.begin()) {
  //  // TODO set user LED to RED
  //  while (1)
  //    delay(10);
  //}
  //as5600.enableWatchdog(false);
  //as5600.setPowerMode(AS5600_POWER_MODE_NOM);
  //as5600.setHysteresis(AS5600_HYSTERESIS_OFF);
  //as5600.setSlowFilter(AS5600_SLOW_FILTER_16X);
  //as5600.setFastFilterThresh(AS5600_FAST_FILTER_THRESH_SLOW_ONLY);
  //as5600.setZPosition(0);
  //as5600.setMPosition(4095);
  //as5600.setMaxAngle(4095);

  // TODO set user LED to GREEN

  // setup serial
  Serial.begin(921600);  // TODO double check this is needed

  static struct repeating_timer controlTimer;
  add_repeating_timer_us(100, motorControllerCallback, NULL, &controlTimer); // 10 kHz low level control for the motor
}


// GLOBAL VARS
volatile unsigned long lastCommandReceived = 0;

// sanitized:
volatile uint16_t motor_command;
volatile bool direction = false; // true = CCW, false = CW

// user commanded
volatile uint16_t user_motor_command;
volatile bool user_direction = false;

volatile int32_t motorEncoderValue;

volatile bool hitLimit = false;


bool motorControllerCallback(struct repeating_timer *t) {
  //if (millis() - lastCommandReceived > COMMAND_TIMEOUT) {
  //  processMotorCommand(0, true);  // kill motor
  //  return true;
  //}

  motorEncoderValue = analogRead(MOTOR_ENC);                                 // TODO is actually 10 bits, switch to 16bits?
  motorEncoderValue = pModulo(motorEncoderValue - MOTOR_OFFSET, MOTOR_CPR);  // [0, CPR]

  // if we hit the limit, we swap and lock the direction
  if (motorEncoderValue > MOTOR_LIMIT && motorEncoderValue < (MOTOR_CPR - MOTOR_LIMIT)) {
    hitLimit = true;
    if (motorEncoderValue > MOTOR_CPR / 2) {
      direction = false;
    } else {
      direction = true;
    }
  }

  float scaling_factor = 1;
  bool outside_limits = (motorEncoderValue < (MOTOR_LIMIT - MOTOR_HYS) || motorEncoderValue > (MOTOR_CPR - MOTOR_LIMIT + MOTOR_HYS));

  // hysteresis before unlocking direction
  if (outside_limits == true) {
    // got outside limits again
    hitLimit = false;
  }

  // only update direction to user command if we did not hit limits, else sticky
  if (hitLimit == false) 
  { 
    direction = user_direction;
  }
  else if (user_direction != direction) { 
    // if we hit the limit and the user still wants to go towards the limit, we make its command 0
    processMotorCommand(0, direction);
    return true;
  }

  if (outside_limits == false) {
    // compute scaling factor, scale only towards the blocking side
    if (motorEncoderValue > (MOTOR_LIMIT - MOTOR_HYS) && user_direction == true) // CCW
    {
      setLED("green");
      scaling_factor = ((MOTOR_LIMIT - MOTOR_HYS)-motorEncoderValue)/MOTOR_HYS;
    }
    
    if (motorEncoderValue < (MOTOR_LIMIT + MOTOR_HYS) && user_direction == false) // CW
    {
      setLED("blue");
      scaling_factor = (motorEncoderValue - (MOTOR_LIMIT - MOTOR_HYS))/MOTOR_HYS;
    }
  }

  motor_command = user_motor_command * scaling_factor;
  processMotorCommand(motor_command, direction);
  return true; // never stop running this callback
}


void loop() {
  if (Serial.available() >= PACKET_SIZE) {
    // check for start sequence
    if (Serial.read() != 0x10) {
      return;
    }
    if (Serial.read() != 0x02) {
      return;
    }

    lastCommandReceived = millis();

    // valid packet, read the command type
    uint8_t command = Serial.read();
    if (command == RESET) { // TODO remove reset command and lower the timeout watchdog, remove command type!
      processMotorCommand(0, true);  // kill motor
      // Discard unnecessary bytes from the serial buffer
      for (int i = 0; i < (PACKET_SIZE - 3); i++) {
        Serial.read();
      }
    } else if (command == STEP) {
      // read motor command into global variables, will be read by routine to actuate
      user_direction = Serial.read();
      Serial.readBytes((char *)&user_motor_command, sizeof(user_motor_command));

      int32_t pendulumEncoderValue = 0;  // as5600.getRawAngle(); // is 12 bits
      unsigned long timestamp = micros();

      Serial.write((uint8_t *)&motorEncoderValue, sizeof(motorEncoderValue));
      Serial.write((uint8_t *)&pendulumEncoderValue, sizeof(pendulumEncoderValue));
      Serial.write((uint8_t *)&timestamp, sizeof(timestamp));
    }
  }
}
