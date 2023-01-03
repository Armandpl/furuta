#include <stdint.h>
#include <Encoder.h>

const uint8_t MOTOR_DRIVER = 0;  // 0 = MC33926
const uint8_t MIN1 = 7;
const uint8_t MIN2 = 8;
const uint8_t MD2 = 6;

// uint8_t MOTOR_DRIVER = 1; // 1 = TB9051-FTG
uint8_t PWM1 = NULL;
uint8_t PWM2 = NULL;

// motor encoder
const uint8_t MOTOR_ENCODER_A = 3;
const uint8_t MOTOR_ENCODER_B = 4;
const float MOTOR_ENCODER_CPR = 211.2;

// pendulum encoder
const uint8_t PENDULUM_ENCODER_A = 2;
const uint8_t PENDULUM_ENCODER_B = 5;
const float PENDULUM_ENCODER_CPR = 8192;

// protocol
const int16_t RESET_ENCODERS = 1000;
const int16_t KILL_MOTOR = 2000;
const uint8_t ACK = 0x55;


void processMotorCommand(int8_t motor_command) {
  if (MOTOR_DRIVER == 0) {  // MC33926
    if (motor_command > 0) {
      digitalWrite(MIN1, LOW);
      digitalWrite(MIN2, HIGH);
    } else {
      digitalWrite(MIN1, HIGH);
      digitalWrite(MIN2, LOW);
    }
    analogWrite(MD2, abs(motor_command));
  } else if (MOTOR_DRIVER == 1) {  // TB9051-FTG
    if (motor_command > 0) {
      analogWrite(PWM1, abs(motor_command));
      analogWrite(PWM2, 0);
    } else {
      analogWrite(PWM1, 0);
      analogWrite(PWM2, abs(motor_command));
    }
  }
}

// Encoder motorEncoder(MOTOR_ENCODER_A, MOTOR_ENCODER_B);
// Encoder pendulumEncoder(PENDULUM_ENCODER_A, PENDULUM_ENCODER_B);

void setup() {
  // setup motor pins
  if (MOTOR_DRIVER == 0) {
    pinMode(MIN1, OUTPUT);
    pinMode(MIN2, OUTPUT);
    pinMode(MD2, OUTPUT);
    analogWrite(MD2, 0);  // check that's needed
  } else {
    pinMode(PWM1, OUTPUT);
    pinMode(PWM2, OUTPUT);
    analogWrite(PWM1, 0);  // check that's needed
    analogWrite(PWM2, 0);
  }

  // setup serial
  //Serial.begin(921600);
  // while (!Serial) {}
  // Serial.setTimeout(10);
}


void loop() {
  // wait until we get one whole dummy packet
  if (Serial.available() > 7) {
    // read dummy packet
    unsigned long start = micros();
    for (int i = 0; i < 8; i++) {
      Serial.read();
    }
    unsigned long elapsed = micros() - start;
    Serial.print("8x Serial.read() = ");
    Serial.print(elapsed);
    Serial.println("us");

    // send back dummy packet
    start = micros();
    uint8_t dummy_bit = 0;
    for (int i = 0; i < 8; i++) {
      Serial.write(dummy_bit);
    }
    elapsed = micros() - start;
    Serial.print("8x Serial.write() = ");
    Serial.print(elapsed);
    Serial.println("us");
  }
}


// void loop() {
//   if (Serial.available() > 3)
//   {
//     for (int i = 0; i< 4; i++) {
//       char c = Serial.read();
//     }

//     for (int i = 0; i < 16; i++) {
//       Serial.write(0);
//     }
//   }
// if (Serial.available() > 1) {
//   // read two bytes into a 16 bit integer using readBytes
//   int16_t command;
//   Serial.readBytes((uint8_t*)&command, 2);

//   // float motor_angle = motorEncoder.read() / MOTOR_ENCODER_CPR * 2 * PI;
//   // float pendulum_angle = pendulumEncoder.read() / PENDULUM_ENCODER_CPR * 2 * PI;

//   // write 8 empty bytes to serial using Serial write
//   for (int i = 0; i < 8; i++) {
//     Serial.write(0);
//   }


// Serial.write((uint8_t*)&motor_angle, sizeof(motor_angle));
// Serial.write((uint8_t*)&pendulum_angle, sizeof(pendulum_angle));

// if (command == 1000) {
//   // motorEncoder.write(0);
//   // pendulumEncoder.write(0);
//   Serial.write(ACK);
// }
// else if (command == 2000) {
//   processMotorCommand(0);
//   uint8_t ack = 0x55;
//   Serial.write(ack);
// }
// else {
//   // processMotorCommand(command);

//   // read system state
//   float motor_angle = motorEncoder.read() / MOTOR_ENCODER_CPR * 2 * PI;
//   float pendulum_angle = pendulumEncoder.read() / PENDULUM_ENCODER_CPR * 2 * PI;

//   // send system state, 4 bytes for each float
//   Serial.write((uint8_t*)&motor_angle, sizeof(motor_angle));
//   Serial.write((uint8_t*)&pendulum_angle, sizeof(pendulum_angle));
// }
// }
// }
