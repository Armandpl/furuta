#include <stdint.h>
#include <Encoder.h>


const uint8_t MOTOR_DRIVER = 0; // 0 = MC33926
const uint8_t MIN1 = 7;
const uint8_t MIN2 = 8;
const uint8_t MD2 = 6;

// uint8_t MOTOR_DRIVER = 1 // 1 = TB9051-FTG
// uint8_t PWM1 = 0;
// uint8_t PWM2 = 0;

// motor encoder
const uint8_t MOTOR_ENCODER_A = 3;
const uint8_t MOTOR_ENCODER_B = 4;
const float MOTOR_ENCODER_CPR = 211.2;

// pendulum encoder
const uint8_t PENDULUM_ENCODER_A = 2;
const uint8_t PENDULUM_ENCODER_B = 5;
const float PENDULUM_ENCODER_CPR = 8192;

// Define the packet structs
struct motor_command_packet {
  uint8_t start_byte; // Start byte to indicate the start of the packet
  uint8_t command_type; // Command type (0 for motor command)
  int16_t motor_command; // Motor command value (-255 to 255)
  uint8_t checksum; // CRC value for packet integrity checking
};

struct read_command_packet {
  uint8_t start_byte; // Start byte to indicate the start of the packet
  uint8_t command_type; // Command type (1 for read command)
  uint8_t checksum; // CRC value for packet integrity checking
};

// answer to read_command
struct system_state_packet {
  uint8_t start_byte; // Start byte to indicate the start of the packet
  uint8_t message_type;
  float motor_angle; // Current motor angle
  float pendulum_angle; // Current pendulum angle
  uint8_t checksum;
}; // TODO add value from current sensor, value from integrated current sensor, value from overheat/overcurrent pin

// struct error_packet {
//   uint8_t start_byte; // Start byte to indicate the start of the packet
//   uint8_t message_type;
//   unint8_t error_code;
//   uint16_t checksum;
// };

// Define the start byte and message types
const uint8_t START_BYTE = 0xAA;
const uint8_t MOTOR_COMMAND = 0x00;
const uint8_t READ_COMMAND = 0x01;
const uint8_t STATE_MESSAGE = 0x02;


/*
  Calculate checksum by XOR-ing all the byte where the pointer "data" points to
  @param data starting address of the data
  @param len length of the data
  @return caculated checksum
*/
uint8_t computeChecksum(void *data, uint8_t len)
{
  uint8_t checksum = 0;
  uint8_t *addr;
  for(addr = (uint8_t*)data; addr < (data + len); addr++){
    // xor all the bytes
    checksum ^= *addr; // checksum = checksum xor value stored in addr
  }
  return checksum;
}

// Function to send a system state packet
void sendSystemState(float motor_angle, float pendulum_angle) {
  // Create a system state packet
  system_state_packet packet;
  packet.start_byte = START_BYTE;
  packet.message_type = STATE_MESSAGE;
  packet.motor_angle = motor_angle;
  packet.pendulum_angle = pendulum_angle;

  // Calculate the CRC value for the packet
  packet.checksum = computeChecksum((uint8_t*)&packet, sizeof(packet) - sizeof(packet.checksum));

  // Send the packet over the serial port
  Serial.write((uint8_t*)&packet, sizeof(packet));
}

void processMotorCommand(int8_t motor_command) {
  if (MOTOR_DRIVER == 0) { // MC33926
    if (motor_command > 0) {
      digitalWrite(MIN1, LOW);
      digitalWrite(MIN2, HIGH);
    }
    else {
      digitalWrite(MIN1, HIGH);
      digitalWrite(MIN2, LOW);
    }
    analogWrite(MD2, abs(motor_command));
  }
  else if (MOTOR_DRIVER == 1) { // TB9051-FTG
    // if (motor_command > 0) {
    //   analogWrite(PWM1, abs(motor_command));
    //   analogWrite(PWM2, 0);
    // }
    // else {
    //   analogWrite(PWM1, 0);
    //   analogWrite(PWM2, abs(motor_command));
    // }
  }
}

Encoder motorEncoder(MOTOR_ENCODER_A, MOTOR_ENCODER_B);
Encoder pendulumEncoder(PENDULUM_ENCODER_A, PENDULUM_ENCODER_B);

// Function to process incoming packets
void processPackets() {
  // Check if any data is available on the serial port
  if (Serial.available() > 0) {
    // Read the next byte from the serial port
    uint8_t b = Serial.read();

    // Serial.println("byte received");
    // Serial.println();
    // Serial.write(&b, 1);
    // Serial.println();

    // Check if the byte is the start byte

    if (b == START_BYTE) {
      // Start byte received, read the next byte to determine the command type
      while (Serial.available() < 1) {} // wait until we have the command type
      uint8_t command_type = Serial.read();
      
      // Check the command type
      if (command_type == MOTOR_COMMAND) {

        // Motor command packet received, read the rest of the packet
        motor_command_packet packet;
        packet.start_byte = b;
        packet.command_type = command_type;

        // remainingpacket size
        // = packet size minus START_BYTE and command_type
        uint8_t remaining_packet_size = sizeof(packet) - sizeof(START_BYTE) - sizeof(command_type);

        // wait until we have the whole packet
        while (Serial.available() < remaining_packet_size) {}

        Serial.readBytes((uint8_t*)&packet.motor_command, remaining_packet_size);

        // Check the packet integrity using the CRC value
        if (computeChecksum((uint8_t*)&packet, sizeof(packet) - sizeof(packet.checksum)) == packet.checksum) {
          // Packet integrity is OK, process the motor command
          processMotorCommand(packet.motor_command);

          // TODO send back ACK?
        }
        else {
          // TODO send back NACK?
        }
      }
      else if (command_type == READ_COMMAND) {
        // Read command packet received, read the rest of the packet
        read_command_packet packet;
        packet.start_byte = b;
        packet.command_type = command_type;

        // remainingpacket size
        // = packet size minus START_BYTE and command_type
        uint8_t remaining_packet_size = sizeof(packet) - sizeof(START_BYTE) - sizeof(command_type);

        // wait until we have the whole packet
        while (Serial.available() < remaining_packet_size) {}

        Serial.readBytes((uint8_t*)&packet.checksum, remaining_packet_size);

        // Check the packet integrity using the CRC value
        if (computeChecksum((uint8_t*)&packet, sizeof(packet) - sizeof(packet.checksum)) == packet.checksum) {
          // Packet integrity is OK, read and send system state
          float motor_angle = motorEncoder.read() / MOTOR_ENCODER_CPR;
          float pendulum_angle = pendulumEncoder.read() / PENDULUM_ENCODER_CPR;

          sendSystemState(motor_angle, pendulum_angle);
        }
        else {
          // TODO send back NACK?
        }
      }
    }
  }  
}

void setup() {
  // setup motor pins
  pinMode(MIN1, OUTPUT);
  pinMode(MIN2, OUTPUT);
  pinMode(MD2, OUTPUT);

  analogWrite(MD2, 0);

  // setup serial
  Serial.begin(115200, SERIAL_8N1);
}

void loop() {
  processPackets();
}
