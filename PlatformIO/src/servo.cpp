#include <Arduino.h>
#include <Servo.h>

// Motor A (connected to EN_A, IN1, IN2)
#define EN_A A0
#define IN1 11
#define IN2 10

// Motor B (connected to EN_B, IN3, IN4)
#define EN_B A1
#define IN3 7
#define IN4 8

// Potentiometers
#define POT1 A2
#define POT2 A3

// PWM Frequency configuration (in Hz)
// Servo library uses 50Hz by default, but we'll use analogWrite-compatible pins
// Frequencies to try: 122, 245, 490, 980, 3920, 7840, 15625 Hz
const int PWM_FREQUENCY_HZ = 490;  // Adjust this value to change PWM frequency
const int PWM_PERIOD_US = 1000000 / PWM_FREQUENCY_HZ;  // Calculate period in microseconds

// Servo objects for PWM control on IN pins
Servo servo_IN1;
Servo servo_IN2;
Servo servo_IN3;
Servo servo_IN4;

// Track which pins are attached to servos
bool attached_IN1 = true;
bool attached_IN2 = true;
bool attached_IN3 = true;
bool attached_IN4 = true;

void setup() {
  Serial.begin(115200);  // Initialize serial communication at 115200 baud
  
  // Configure EN pins as outputs (always set to HIGH/255)
  pinMode(EN_A, OUTPUT);
  pinMode(EN_B, OUTPUT);
  digitalWrite(EN_A, HIGH);
  digitalWrite(EN_B, HIGH);
  analogWrite(EN_A, 255);
  analogWrite(EN_B, 255);
  
  // Attach servo objects to IN pins for PWM control
  servo_IN1.attach(IN1);
  servo_IN2.attach(IN2);
  servo_IN3.attach(IN3);
  servo_IN4.attach(IN4);
  
  // Configure potentiometers as inputs
  pinMode(POT1, INPUT);
  pinMode(POT2, INPUT);
  
  // Initialize motors to stop (0% duty cycle - pins LOW, not attached to servo)
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

void setMotor(Servo &in1, Servo &in2, int pwmValue, bool &attached1, bool &attached2, int pin1, int pin2) {
  // pwmValue ranges from -255 to 255
  // Maps directly to duty cycle percentage
  // Positive: IN1 gets PWM, IN2 is LOW (0V)
  // Negative: IN1 is LOW (0V), IN2 gets PWM
  // 0: both LOW (0% duty)
  // 255/-255: 100% duty on respective pin
  
  int pwmValue_constrained = constrain(pwmValue, -255, 255);
  
  if (pwmValue_constrained == 0) {
    // Stop: Both pins directly to LOW (no servo control)
    if (attached1) {
      in1.detach();
      attached1 = false;
    }
    if (attached2) {
      in2.detach();
      attached2 = false;
    }
    pinMode(pin1, OUTPUT);
    pinMode(pin2, OUTPUT);
    digitalWrite(pin1, LOW);
    digitalWrite(pin2, LOW);
  } else {
    // Reattach servos if they were detached
    if (!attached1) {
      in1.attach(pin1);
      attached1 = true;
    }
    if (!attached2) {
      in2.attach(pin2);
      attached2 = true;
    }
    
    // Calculate pulse width based on duty cycle
    // 0-255 maps to 0-100% duty cycle
    int onTime = map(abs(pwmValue_constrained), 0, 255, 0, PWM_PERIOD_US);
    
    if (pwmValue_constrained > 0) {
      // Forward: IN1 gets PWM, IN2 is LOW
      in1.writeMicroseconds(onTime);
      in2.writeMicroseconds(0);
    } else {
      // Reverse: IN1 is LOW, IN2 gets PWM
      in1.writeMicroseconds(0);
      in2.writeMicroseconds(onTime);
    }
  }
}

void loop() {
  // Read potentiometer values
  int pot1Value = analogRead(POT1);
  int pot2Value = analogRead(POT2);
  
  // Print potentiometer values
  Serial.print("Pot1: ");
  Serial.print(pot1Value);
  Serial.print(" | Pot2: ");
  Serial.println(pot2Value);

  if (Serial.available() > 0) {
    // Read the input string until newline
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    // Parse the two integers
    int spaceIndex = input.indexOf(' ');
    
    if (spaceIndex > 0) {
      int pwm1 = input.substring(0, spaceIndex).toInt();
      int pwm2 = input.substring(spaceIndex + 1).toInt();
      
      // Constrain values to -255 to 255
      pwm1 = constrain(pwm1, -255, 255);
      pwm2 = constrain(pwm2, -255, 255);
      
      // Set motor speeds
      setMotor(servo_IN1, servo_IN2, pwm1, attached_IN1, attached_IN2, IN1, IN2);
      setMotor(servo_IN3, servo_IN4, pwm2, attached_IN3, attached_IN4, IN3, IN4);
      
      // Echo back the values for confirmation
      Serial.print("Motor A: ");
      Serial.print(pwm1);
      Serial.print(" | Motor B: ");
      Serial.print(pwm2);
      Serial.print(" | PWM Freq: ");
      Serial.print(PWM_FREQUENCY_HZ);
      Serial.println(" Hz");
    }
  }
  
  delay(100);  // Print potentiometer values every 100ms
}