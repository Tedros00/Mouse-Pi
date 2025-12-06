#include <Arduino.h>

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

void setup() {
  Serial.begin(115200);  // Initialize serial communication at 115200 baud
  
  // Configure motor control pins as outputs
  pinMode(EN_A, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  
  pinMode(EN_B, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  
  // Configure potentiometers as inputs
  pinMode(POT1, INPUT);
  pinMode(POT2, INPUT);
  
  // Initialize motors to stop
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  analogWrite(EN_A, 0);
  analogWrite(EN_B, 0);
}

void setMotor(int pwmPin, int in1Pin, int in2Pin, int dutyCycle) {
  // dutyCycle ranges from -100 to 100
  // Positive values: IN1 HIGH, IN2 LOW
  // Negative values: IN1 LOW, IN2 HIGH
  
  int pwmValue = abs(dutyCycle) * 255 / 100;  // Convert duty cycle to 0-255
  pwmValue = constrain(pwmValue, 0, 255);
  
  if (dutyCycle > 0) {
    // Forward direction
    digitalWrite(in1Pin, HIGH);
    digitalWrite(in2Pin, LOW);
  } else if (dutyCycle < 0) {
    // Reverse direction
    digitalWrite(in1Pin, LOW);
    digitalWrite(in2Pin, HIGH);
  } else {
    // Stop
    digitalWrite(in1Pin, LOW);
    digitalWrite(in2Pin, LOW);
  }
  
  analogWrite(pwmPin, pwmValue);
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
      int dc1 = input.substring(0, spaceIndex).toInt();
      int dc2 = input.substring(spaceIndex + 1).toInt();
      
      // Constrain values to -100 to 100
      dc1 = constrain(dc1, -100, 100);
      dc2 = constrain(dc2, -100, 100);
      
      // Set motor speeds
      setMotor(EN_A, IN1, IN2, dc1);
      setMotor(EN_B, IN3, IN4, dc2);
      
      // Echo back the values for confirmation
      Serial.print("Motor A: ");
      Serial.print(dc1);
      Serial.print("% | Motor B: ");
      Serial.print(dc2);
      Serial.println("%");
    }
  }
  
  delay(100);  // Print potentiometer values every 100ms
}