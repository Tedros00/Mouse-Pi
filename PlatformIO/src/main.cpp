#include <Arduino.h>

// Motor A (connected to EN_A, IN1, IN2)
#define EN_A A0
#define IN1 11
#define IN2 10

// Motor B (connected to EN_B, IN3, IN4)
#define EN_B A1
#define IN3 7
#define IN4 8

// PWM Frequency control
// Lowering frequency gives more torque at low speeds but may cause buzzing
// Higher frequency is smoother but may need higher minimum PWM
#define PWM_FREQUENCY 490  // Default Arduino frequency; try 122, 245, 490, 980, 3920, 7840, 15625 Hz

// Potentiometers
#define POT1 A2
#define POT2 A3

// Forward declaration of PWM frequency function
void setPwmFrequency(int pin, int divisor);

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
  
  // Set PWM frequency for motor pins (pins 9 and 10 share timer1, 3 and 11 share timer2)
  // For pins 11 (IN2/EN_A) and 3 (IN4/EN_B): change timer2 frequency
  setPwmFrequency(11, 1);  // Divider 1 = 31.4 kHz (too high, audible noise reduced)
  setPwmFrequency(3, 1);   // Divider 1 = 31.4 kHz
  
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
  
  // Define minimum and maximum PWM values for motor control
  // MIN_PWM: minimum speed threshold to start motor rotation (adjust to your motor's needs)
  // MAX_PWM: typically 255, but can be reduced to limit maximum speed
  const int MIN_PWM = 36;    // You found motors activate at 36
  const int MAX_PWM = 255;
  
  // Map duty cycle to PWM range with dead band
  int pwmValue = 0;
  if (dutyCycle != 0) {
    // Map 1-100% to MIN_PWM-MAX_PWM range
    pwmValue = MIN_PWM + (abs(dutyCycle) - 1) * (MAX_PWM - MIN_PWM) / 99;
    pwmValue = constrain(pwmValue, MIN_PWM, MAX_PWM);
  }
  
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

// Set PWM frequency on Arduino pins
// Divisor: 1=31.4kHz, 2=3.9kHz, 3=980Hz, 4=490Hz, 5=245Hz, 6=122Hz, 7=61Hz
void setPwmFrequency(int pin, int divisor) {
  byte mode;
  if(pin == 5 || pin == 6 || pin == 9 || pin == 10) {
    switch(divisor) {
      case 1: mode = 0x01; break;
      case 2: mode = 0x02; break;
      case 3: mode = 0x03; break;
      case 4: mode = 0x04; break;
      case 5: mode = 0x05; break;
      case 6: mode = 0x06; break;
      case 7: mode = 0x07; break;
      default: return;
    }
    TCCR1B = (TCCR1B & 0b11111000) | mode;
  }
  else if(pin == 3 || pin == 11) {
    switch(divisor) {
      case 1: mode = 0x01; break;
      case 2: mode = 0x02; break;
      case 3: mode = 0x03; break;
      case 4: mode = 0x04; break;
      case 5: mode = 0x05; break;
      case 6: mode = 0x06; break;
      case 7: mode = 0x07; break;
      default: return;
    }
    TCCR2B = (TCCR2B & 0b11111000) | mode;
  }
}