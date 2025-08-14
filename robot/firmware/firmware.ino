#include <stdint.h>
#include <Encoder.h>

#define PI 3.1415926535897932384626433832795

const float MOTOR_CPR = 400.0;
const float PENDULUM_CPR = 5120.0 * 4;

// protocol def
const uint8_t PACKET_SIZE = 11;
const uint8_t RESET = 0;
const uint8_t STEP = 1;
const uint8_t STEP_PID = 2;

// Pin definitions
// motor driver
uint8_t PWM1 = 2;
uint8_t PWM2 = 3;

const uint8_t MOTOR_ENC_A = 9;
const uint8_t MOTOR_ENC_B = 10;

const uint8_t PENDULUM_ENC_A = 7;
const uint8_t PENDULUM_ENC_B = 8;

Encoder motorEncoder(MOTOR_ENC_A, MOTOR_ENC_B);
Encoder pendulumEncoder(PENDULUM_ENC_A, PENDULUM_ENC_B);

// Watchdog
volatile unsigned long lastCommandReceived = 0;
const unsigned long COMMAND_TIMEOUT = 500; // ms
bool watchdogTriggered = true;

uint8_t commandType = RESET;

float motorDesiredPosition = 0.0;
float motorDesiredVelocity = 0.0;
bool motorDirection = false;
uint16_t motorCommand = 0;
float pdCommand = 0;

float motorPosition = 0.0;
float pendulumPosition = 0.0;
float motorVelocity = 0.0;
float pendulumVelocity = 0.0;
unsigned long timestamp = 0.0;

float Kp = 4.0;
float Kv = 0.0;
float TAU = 0.03;

void reset() {
  processMotorCommand(0, true); // kill motor
  motorEncoder.write(0);
  pendulumEncoder.write(0);
  motorDesiredPosition = 0.0;
  motorDesiredVelocity = 0.0;
  motorCommand = 0;
  watchdogTriggered = false;
}

void processMotorCommand(uint16_t motorCommand, bool direction) {
  if (direction) {
    analogWrite(PWM1, motorCommand);
    analogWrite(PWM2, 0);
  }
  else {
    analogWrite(PWM1, 0);
    analogWrite(PWM2, motorCommand);
  }
}


void setup() {
  // setup motor pins
  pinMode(PWM1, OUTPUT);
  pinMode(PWM2, OUTPUT);

  // Apparently the PWM resolution on the SAMD21 is 16 bits
  // https://tomalmy.com/analogwriteresolution-and-the-samd21/
  analogWriteResolution(16);

  // setup serial
  Serial.begin(921600);
}


void loop() {
  // check for new message
  if (Serial.available() >= PACKET_SIZE)
  {
    // check for start sequence
    if(Serial.read() != 0x10){
      return;
    }
    if(Serial.read() != 0x02){
      return;
    }
    // valid packet
    lastCommandReceived = millis();

    // read the command type
    commandType = Serial.read();
    uint8_t bytesRead = 3;
    if (commandType == RESET)
    {
      reset();
    }
    else if (commandType == STEP)
    {
      motorDirection = Serial.read();
      Serial.readBytes((char*)&motorCommand, sizeof(motorCommand));
      bytesRead += 3;
    }
    else if (commandType == STEP_PID)
    {
      Serial.readBytes((char*)&motorDesiredPosition, sizeof(motorDesiredPosition));
      Serial.readBytes((char*)&motorDesiredVelocity, sizeof(motorDesiredVelocity));
      bytesRead += 8;
    }
    // Discard unnecessary bytes from the serial buffer
    for (int i = 0; i < (PACKET_SIZE - bytesRead); i++)
    {
      Serial.read();
    }

    Serial.write((uint8_t*)&motorPosition, sizeof(motorPosition));
    Serial.write((uint8_t*)&pendulumPosition, sizeof(pendulumPosition));
    Serial.write((uint8_t*)&motorVelocity, sizeof(motorVelocity));
    Serial.write((uint8_t*)&pendulumVelocity, sizeof(pendulumVelocity));
    Serial.write((uint8_t*)&timestamp, sizeof(timestamp));
    Serial.write((uint8_t*)&motorDirection, sizeof(motorDirection));
    Serial.write((uint8_t*)&motorCommand, sizeof(motorCommand));
  }

  // Watchdog
  if (millis() - lastCommandReceived > COMMAND_TIMEOUT) {
    processMotorCommand(0, true); // kill motor
    watchdogTriggered = true;
    return;
  }

  // Read encoders
  float motorPositionNew = motorEncoder.read() * 2 * PI / MOTOR_CPR;
  float pendulumPositionNew = pendulumEncoder.read() * 2 * PI / PENDULUM_CPR;
  unsigned long timestampNew = micros();

  float dt = (timestampNew - timestamp) * 1e-6;

  // Estimate velocity via finite differences and discrete low pass filter
  motorVelocity = (2 * (motorPositionNew - motorPosition) / dt - (1 - 2 * TAU / dt) * motorVelocity) / (1 + 2 * TAU / dt);
  pendulumVelocity = (2 * (pendulumPositionNew - pendulumPosition) / dt - (1 - 2 * TAU / dt) * pendulumVelocity) / (1 + 2 * TAU / dt);

  motorPosition = motorPositionNew;
  pendulumPosition = pendulumPositionNew;
  timestamp = timestampNew;

  if (commandType == STEP_PID)
  {
    pdCommand = Kp * (motorDesiredPosition - motorPosition) + Kv * (motorDesiredVelocity - motorVelocity);
    motorDirection = (pdCommand < 0.0);
    motorCommand = constrain(abs(pdCommand), 0.0, 1.0) * UINT16_MAX;
  }

  if (not watchdogTriggered)
  {
    // process the motor command
    processMotorCommand(motorCommand, motorDirection);
  }
}
