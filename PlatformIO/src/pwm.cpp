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

// ============================================================================
// Hardware PWM Interrupt-Based Control
// ============================================================================

// PWM Configuration
const uint16_t PWM_FREQUENCY_HZ = 10000;           // PWM frequency in Hz

// PWM state for each motor (0-255, where 0=0% duty, 255=100% duty)
volatile uint8_t pwm_motor_a = 0;  // Motor A PWM value (0-255)
volatile uint8_t pwm_motor_b = 0;  // Motor B PWM value (0-255)

// PWM direction control (true = positive direction, false = negative direction)
volatile bool dir_motor_a = true;   // true: IN1 gets PWM, false: IN2 gets PWM
volatile bool dir_motor_b = true;   // true: IN3 gets PWM, false: IN4 gets PWM

// Internal timing state for interrupt handler
volatile uint16_t pwm_timer_count = 0;  // Current timer tick count within PWM period

// ============================================================================
// Timer1 Interrupt Handler - Runs at 16kHz (62.5us intervals)
// Generates PWM signals based on current pwm_motor_a and pwm_motor_b values
// ============================================================================
ISR(TIMER1_COMPA_vect) {
  // Motor A control: EN_A gets PWM, IN1/IN2 set direction
  if (pwm_timer_count <= pwm_motor_a) {
    digitalWrite(EN_A, HIGH);  // Enable motor (PWM high)
  } else {
    digitalWrite(EN_A, LOW);   // Disable motor (PWM low)
  }
  
  // Set direction for Motor A
  if (dir_motor_a) {
    digitalWrite(IN1, HIGH);   // Forward direction
    digitalWrite(IN2, LOW);
  } else {
    digitalWrite(IN1, LOW);    // Reverse direction
    digitalWrite(IN2, HIGH);
  }
  
  // Motor B control: EN_B gets PWM, IN3/IN4 set direction
  if (pwm_timer_count <= pwm_motor_b) {
    digitalWrite(EN_B, HIGH);  // Enable motor (PWM high)
  } else {
    digitalWrite(EN_B, LOW);   // Disable motor (PWM low)
  }
  
  // Set direction for Motor B
  if (dir_motor_b) {
    digitalWrite(IN3, HIGH);   // Forward direction
    digitalWrite(IN4, LOW);
  } else {
    digitalWrite(IN3, LOW);    // Reverse direction
    digitalWrite(IN4, HIGH);
  }
  
  // Increment timer counter and reset at period end
  pwm_timer_count++;
  if (pwm_timer_count >= 256) {
    pwm_timer_count = 0;
  }
}

// ============================================================================
// Public API Functions
// ============================================================================

/**
 * Initialize Timer1 for PWM interrupt-based control
 * Sets up CTC mode with interrupts at 16kHz
 */
void pwm_init() {
  // Timer1 Configuration for CTC mode
  TCCR1A = 0;                     // Normal mode
  TCCR1B = 0;                     // Reset
  TCNT1 = 0;                      // Reset counter
  
  // Set CTC mode (WGM12 = 1)
  TCCR1B |= (1 << WGM12);
  
  // Set prescaler to 1 (no prescaling) for 16MHz -> 16MHz ticks
  TCCR1B |= (1 << CS10);
  
  // Set compare value for 16kHz interrupt (16000000 / 16000 = 1000 ticks)
  OCR1A = 1000;
  
  // Enable Timer1 Compare A interrupt
  TIMSK1 |= (1 << OCIE1A);
}

/**
 * Set PWM value for Motor A
 * @param pwm_value: 0-255 (0% to 100% duty cycle)
 *        Sign determines direction: positive -> IN1 active, negative -> IN2 active
 */
void setPWM_MotorA(int8_t pwm_value) {
  if (pwm_value >= 0) {
    dir_motor_a = true;              // IN1 carries PWM signal
    pwm_motor_a = constrain(pwm_value, 0, 255);
  } else {
    dir_motor_a = false;             // IN2 carries PWM signal
    pwm_motor_a = constrain(-pwm_value, 0, 255);
  }
}

/**
 * Set PWM value for Motor B
 * @param pwm_value: 0-255 (0% to 100% duty cycle)
 *        Sign determines direction: positive -> IN3 active, negative -> IN4 active
 */
void setPWM_MotorB(int8_t pwm_value) {
  if (pwm_value >= 0) {
    dir_motor_b = true;              // IN3 carries PWM signal
    pwm_motor_b = constrain(pwm_value, 0, 255);
  } else {
    dir_motor_b = false;             // IN4 carries PWM signal
    pwm_motor_b = constrain(-pwm_value, 0, 255);
  }
}

/**
 * Stop both motors immediately
 */
void stopAllMotors() {
  pwm_motor_a = 0;
  pwm_motor_b = 0;
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

// PWM Frequency control
void setup() {
  Serial.begin(115200);  // Initialize serial communication at 115200 baud
  
  // Initialize hardware PWM interrupt system
  pwm_init();
  
  // Configure EN pins as outputs (controlled by interrupt)
  pinMode(EN_A, OUTPUT);
  pinMode(EN_B, OUTPUT);
  digitalWrite(EN_A, LOW);
  digitalWrite(EN_B, LOW);
  
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

void loop() {
  if (Serial.available() > 0) {
    // Read the input character or string
    char inputChar = Serial.read();
    
    // Check if 'R' was received to send potentiometer values
    if (inputChar == 'R' || inputChar == 'r') {
      int pot1Value = analogRead(POT1);
      int pot2Value = analogRead(POT2);
      
      Serial.print(pot1Value);
      Serial.print(" ");
      Serial.println(pot2Value);
    }
    // Check if it's a digit (start of PWM command)
    else if (isdigit(inputChar) || inputChar == '-') {
      // Read the rest of the line
      String input = String(inputChar) + Serial.readStringUntil('\n');
      input.trim();
      
      // Parse the two integers
      int spaceIndex = input.indexOf(' ');
      
      if (spaceIndex > 0) {
        int dc1 = input.substring(0, spaceIndex).toInt();
        int dc2 = input.substring(spaceIndex + 1).toInt();
        
        // Map -100 to 100 range to -255 to 255 for PWM
        dc1 = constrain(dc1, -255, 255);
        dc2 = constrain(dc2, -255, 255);
        
        // Convert duty cycle percentage to PWM value (0-255)
        int pwm1 = map(dc1, -255, 255, -100, 100);
        int pwm2 = map(dc2, -255, 255, -100, 100);
        
        // Set motor PWM values
        setPWM_MotorA(pwm1);
        setPWM_MotorB(pwm2);
      }
    }
  }
}
  


