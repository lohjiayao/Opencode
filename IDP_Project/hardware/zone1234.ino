#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
#define SERVO_FREQ 50

// --- ZONE 2 & 3 SERVO CHANNELS (PCA9685) ---
const int SERVO_G        = 5;    // 270 Standard Positional Servo (Gripper Wrist Alignment)
const int NEW_360_SERVO  = 6;    // Continuous Gripper Jaws Rotator (Open/Close)

// Global True Zero reference stops for your 360-degree continuous servos
int stopPulsesMap[7] = {1600, 1600, 1600, 1600, 1600, 1600, 1600};

// --- STEPPER MOTOR PIN CONFIGURATIONS ---
const int STEP_B_PUL  = 4;       // Y-Axis 1 Pulse Pin (Long Vertical Axis)
const int STEP_B_DIR  = 5;       // Y-Axis 1 Direction Pin
const int STEP_C_PUL  = 6;       // X-Axis Pulse Pin (Horizontal Axis)
const int STEP_C_DIR  = 7;       // X-Axis Direction Pin
const int STEP_D_PUL  = 8;       // Lead Screw Axis (Vertical Lift)
const int STEP_D_DIR  = 11;      // Lead Screw Direction Pin
const int STEP_Y2_PUL = 12;      // Y-Axis 2 Pulse Pin (Parallel Motor)
const int STEP_Y2_DIR = 13;      // Y-Axis 2 Direction Pin

// --- MACHINE CALIBRATION PARAMETERS ---
const long STEPPER_33CM_PULSES  = 13500; // Baseline Zone 2 Gantry Travel Distance
const long STEPPER_102MM_PULSES = 4080;  // Conveyor centering cross travel width
const long STEPPER_9CM_PULSES   = 18000; // Corrected Lead screw 9cm drop distance
const int WAVE_DELAY_US         = 600;   // Stepper pulse timing frequency delay

// Gripper 360 Open/Close Parameters
const int OUTWARD_VALUE         = 1500;  
const int INWARD_VALUE          = 1700;  
const int GRIPPER_OPEN_DURATION = 560;  
const int GRIPPER_CLOS_DURATION = 590;

// Positional calibration tracking engine map for 270-degree servo 5
const int SERVO_5_0_DEG         = 2600;
const int SERVO_5_180_DEG       = 500;  

// --- ZONE 3 3x3 MATRIX CALIBRATION (10cm Steps) ---
const long POS_0 = 0;            // 0cm Marker
const long POS_1 = 4000;         // 10cm Marker
const long POS_2 = 8000;         // 20cm Marker

long currentX = 0;               // Persistent tracking engine register for X
long currentY = 0;               // Persistent tracking engine register for Y

// Grid Matrix Slots Counters
int prod1_count = 0;             // Row Y0 Slot counter (Product 1)
int prod2_count = 0;             // Row Y1 Slot counter (Product 2)
int wrong_count = 0;             // Row Y2 Slot counter (Rejections)

int globalCalculatedAngle = 0;   // Dynamic holding registry for Python orientation metrics
bool machineActive = false;      // Master system run flag state

// Forward Declarations
void runZone2Sequence();
void runDynamicZone3Sequence(int productType);
void moveToCoordinate(long targetX, long targetY);
void processGridPoint(long xPos, long yPos);
void enforceAllRestingStops();

void setup() {
  Serial.begin(9600);
  Wire.begin();
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);
  
  // IMMEDIATELY isolate the upgrade continuous gripper to prevent spin loops at startup
  pwm.writeMicroseconds(NEW_360_SERVO, 1600);
  enforceAllRestingStops();
  
  // Home wrist assembly to absolute neutral 0 degrees straight orientation
  pwm.writeMicroseconds(SERVO_G, SERVO_5_0_DEG);

  // Initialize Stepper Output Driver Pins
  pinMode(STEP_B_PUL, OUTPUT);  pinMode(STEP_B_DIR, OUTPUT);
  pinMode(STEP_C_PUL, OUTPUT);  pinMode(STEP_C_DIR, OUTPUT);
  pinMode(STEP_D_PUL, OUTPUT);  pinMode(STEP_D_DIR, OUTPUT);
  pinMode(STEP_Y2_PUL, OUTPUT); pinMode(STEP_Y2_DIR, OUTPUT);

  Serial.println("[HARDWARE] Controller initialization complete node secure.");
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "PING") {
      Serial.println("PONG");
    }
    else if (command == "START_INSPECTION_RUN") {
      machineActive = true;
    }
    // --- PARSE ORIENTATION DEVIATION DATA FROM PYTHON CAMERA ---
    else if (command.startsWith("MOVE:")) {
      int firstColon = command.indexOf(':');
      int secondColon = command.indexOf(':', firstColon + 1);
      if (firstColon != -1 && secondColon != -1) {
        String angleStr = command.substring(secondColon + 1);
        float deviation = angleStr.toFloat();
        // 0 deviation equals 0 degrees absolute neutral straight position
        globalCalculatedAngle = constrain((int)(deviation), 0, 180);
      }
    }
    // --- INTEGRATED AUTOMATED ROUTING INTERACTION SYSTEM ---
    else if (command.startsWith("TRIGGER_PROD_1") || command.startsWith("TRIGGER_PROD_2") || command == "TRIGGER_WRONG") {
      machineActive = false; // Halt intake workflows during active pick operations

      if (command == "TRIGGER_WRONG") {
        // Red Box/Wrong Product: Bypass manual alignment step sequences, route straight to Reject bin
        runZone2Sequence();
        runDynamicZone3Sequence(3);
      }
      else {
        // Process standard product alignments (Green, White, Yellow, Orange, Cyan)
        runZone2Sequence();
        if (command.startsWith("TRIGGER_PROD_1")) {
          runDynamicZone3Sequence(1); // Row Y0
        } else {
          runDynamicZone3Sequence(2); // Row Y1
        }
      }
      machineActive = true; // Safe completion; allow next sorting pass
    }
    // --- HARDWARE DYNAMIC ZERO CALIBRATIONS ---
    else if (command.startsWith("SET_ZERO:")) {
      int s1 = command.indexOf(':');
      int s2 = command.indexOf(':', s1 + 1);
      int ch = command.substring(s1+1, s2).toInt();
      int val = command.substring(s2+1).toInt();
      if (ch >= 0 && ch < 7) {
        stopPulsesMap[ch] = val;
        pwm.writeMicroseconds(ch, val);
      }
    }
    else if (command.equals("EMERGENCY_STOP")) {
      enforceAllRestingStops();
    }
  }
}

void enforceAllRestingStops() {
  for (int i = 0; i < 7; i++) pwm.writeMicroseconds(i, stopPulsesMap[i]);
}

// ======================================================================
// --- ZONE 2 ENTRY POINT: EXTEND GANTRY TO PICKUP POSITION -----------
// ======================================================================
void runZone2Sequence() {
  digitalWrite(STEP_B_DIR, LOW);  
  digitalWrite(STEP_Y2_DIR, LOW);  
  digitalWrite(STEP_C_DIR, HIGH);
  
  long pulses_b = 0;
  long pulses_c = 0;
  while (pulses_b < STEPPER_33CM_PULSES || pulses_c < STEPPER_102MM_PULSES) {
    if (pulses_b < STEPPER_33CM_PULSES) {
      digitalWrite(STEP_B_PUL, HIGH);
      digitalWrite(STEP_Y2_PUL, HIGH);
    }
    if (pulses_c < STEPPER_102MM_PULSES) digitalWrite(STEP_C_PUL, HIGH);
   
    delayMicroseconds(WAVE_DELAY_US);
   
    if (pulses_b < STEPPER_33CM_PULSES) {
      digitalWrite(STEP_B_PUL, LOW);
      digitalWrite(STEP_Y2_PUL, LOW);  
      pulses_b++;
    }
    if (pulses_c < STEPPER_102MM_PULSES) { digitalWrite(STEP_C_PUL, LOW); pulses_c++; }
   
    delayMicroseconds(WAVE_DELAY_US);
  }
 
  delay(500);
  
  // Align wrist rotation to python alignment calculation map directly matching home index
  int microsecondsTarget = map(globalCalculatedAngle, 0, 180, SERVO_5_0_DEG, SERVO_5_180_DEG);
  pwm.writeMicroseconds(SERVO_G, microsecondsTarget);
  delay(1000);
  
  // Open jaws to preparation width
  pwm.writeMicroseconds(NEW_360_SERVO, OUTWARD_VALUE);
  delay(GRIPPER_OPEN_DURATION);
  pwm.writeMicroseconds(NEW_360_SERVO, stopPulsesMap[NEW_360_SERVO]);
  delay(1000);
  
  // Drive lead screw down 9cm to secure item
  digitalWrite(STEP_D_DIR, HIGH);
  for (long i = 0; i < STEPPER_9CM_PULSES; i++) {
    digitalWrite(STEP_D_PUL, HIGH); delayMicroseconds(WAVE_DELAY_US);
    digitalWrite(STEP_D_PUL, LOW);  delayMicroseconds(WAVE_DELAY_US);
  }
  delay(1000);
  
  // Clamp item firmly
  pwm.writeMicroseconds(NEW_360_SERVO, INWARD_VALUE);
  delay(GRIPPER_CLOS_DURATION);
  pwm.writeMicroseconds(NEW_360_SERVO, stopPulsesMap[NEW_360_SERVO]);
  delay(1500);
  
  // Lift lead screw 9cm back to safety height
  digitalWrite(STEP_D_DIR, LOW);
  for (long i = 0; i < STEPPER_9CM_PULSES; i++) {
    digitalWrite(STEP_D_PUL, HIGH); delayMicroseconds(WAVE_DELAY_US);
    digitalWrite(STEP_D_PUL, LOW);  delayMicroseconds(WAVE_DELAY_US);
  }
  delay(1000);
  
  // Retract Gantry back home to origin references
  digitalWrite(STEP_B_DIR, HIGH);
  digitalWrite(STEP_Y2_DIR, HIGH);
  digitalWrite(STEP_C_DIR, LOW);  
  
  pulses_b = 0;
  pulses_c = 0;
  while (pulses_b < STEPPER_33CM_PULSES || pulses_c < STEPPER_102MM_PULSES) {
    if (pulses_b < STEPPER_33CM_PULSES) {
      digitalWrite(STEP_B_PUL, HIGH);
      digitalWrite(STEP_Y2_PUL, HIGH);
    }
    if (pulses_c < STEPPER_102MM_PULSES) digitalWrite(STEP_C_PUL, HIGH);
   
    delayMicroseconds(WAVE_DELAY_US);
   
    if (pulses_b < STEPPER_33CM_PULSES) {
      digitalWrite(STEP_B_PUL, LOW);
      digitalWrite(STEP_Y2_PUL, LOW);
      pulses_b++;
    }
    if (pulses_c < STEPPER_102MM_PULSES) { digitalWrite(STEP_C_PUL, LOW); pulses_c++; }
   
    delayMicroseconds(WAVE_DELAY_US);
  }
  currentX = 0;
  currentY = 0;
}

// ======================================================================
// --- ZONE 3 GANTRY COORDINATE XY TRACKING CONTROLLER ------------------
// ======================================================================
void moveToCoordinate(long targetX, long targetY) {
  long stepsToMoveX = targetX - currentX;
  long stepsToMoveY = targetY - currentY;
  
  if (stepsToMoveX >= 0) digitalWrite(STEP_C_DIR, HIGH);
  else {
    digitalWrite(STEP_C_DIR, LOW);
    stepsToMoveX = -stepsToMoveX;  
  }
  
  if (stepsToMoveY >= 0) {
    digitalWrite(STEP_B_DIR, LOW);  
    digitalWrite(STEP_Y2_DIR, LOW);
  } else {
    digitalWrite(STEP_B_DIR, HIGH);  
    digitalWrite(STEP_Y2_DIR, HIGH);
    stepsToMoveY = -stepsToMoveY;  
  }
  
  long pulses_x = 0;
  long pulses_y = 0;
  while (pulses_x < stepsToMoveX || pulses_y < stepsToMoveY) {
    if (pulses_x < stepsToMoveX) digitalWrite(STEP_C_PUL, HIGH);
    if (pulses_y < stepsToMoveY) {
      digitalWrite(STEP_B_PUL, HIGH);
      digitalWrite(STEP_Y2_PUL, HIGH);
    }
   
    delayMicroseconds(WAVE_DELAY_US);
   
    if (pulses_x < stepsToMoveX) { digitalWrite(STEP_C_PUL, LOW); pulses_x++; }
    if (pulses_y < stepsToMoveY) {
      digitalWrite(STEP_B_PUL, LOW);
      digitalWrite(STEP_Y2_PUL, LOW);  
      pulses_y++;
    }
   
    delayMicroseconds(WAVE_DELAY_US);
  }
  currentX = targetX;
  currentY = targetY;
  delay(500);
}

// ======================================================================
// --- ZONE 3 PLACEMENT MACRO MODULE WITH 9CM DROP CONTROL -------------
// ======================================================================
void processGridPoint(long xPos, long yPos) {
  moveToCoordinate(xPos, yPos);
 
  int microsecondsTarget = map(globalCalculatedAngle, 0, 180, SERVO_5_0_DEG, SERVO_5_180_DEG);
  pwm.writeMicroseconds(SERVO_G, microsecondsTarget);
  delay(1000);
  
  // Lower lead screw 9cm into delivery slot
  digitalWrite(STEP_D_DIR, HIGH);
  for (long i = 0; i < STEPPER_9CM_PULSES; i++) {
    digitalWrite(STEP_D_PUL, HIGH); delayMicroseconds(WAVE_DELAY_US);
    digitalWrite(STEP_D_PUL, LOW);  delayMicroseconds(WAVE_DELAY_US);
  }
  delay(1000);
  
  // Open gripper jaws to drop component safely
  pwm.writeMicroseconds(NEW_360_SERVO, OUTWARD_VALUE);
  delay(GRIPPER_OPEN_DURATION);
  pwm.writeMicroseconds(NEW_360_SERVO, stopPulsesMap[NEW_360_SERVO]);
  delay(1000);
  
  // Retract lead screw 9cm back up to baseline safety height
  digitalWrite(STEP_D_DIR, LOW);
  for (long i = 0; i < STEPPER_9CM_PULSES; i++) {
    digitalWrite(STEP_D_PUL, HIGH); delayMicroseconds(WAVE_DELAY_US);
    digitalWrite(STEP_D_PUL, LOW);  delayMicroseconds(WAVE_DELAY_US);
  }
  delay(1000);
}

// ======================================================================
// --- CHUTE ROUTER STRATEGY SCHEDULER ---------------------------------
// ======================================================================
void runDynamicZone3Sequence(int productType) {
  long targetX = POS_0;
  long targetY = POS_0;
 
  if (productType == 1) { // White Box (Product 1) -> Row Y0
    targetY = POS_0; 
    if (prod1_count == 0)      targetX = POS_0;
    else if (prod1_count == 1) targetX = POS_1;
    else                       targetX = POS_2;
   
    processGridPoint(targetX, targetY);
    prod1_count = (prod1_count + 1) % 3; 
  }
  else if (productType == 2) { // Green / Yellow Box (Product 2) -> Row Y1
    targetY = POS_1; 
    if (prod2_count == 0)      targetX = POS_0;
    else if (prod2_count == 1) targetX = POS_1;
    else                       targetX = POS_2;
   
    processGridPoint(targetX, targetY);
    prod2_count = (prod2_count + 1) % 3;
  }
  else if (productType == 3) { // Red Box (Rejections) -> Row Y2 Reject Chute
    targetY = POS_2; 
    if (wrong_count == 0)      targetX = POS_0;
    else if (wrong_count == 1) targetX = POS_1;
    else                       targetX = POS_2;
   
    processGridPoint(targetX, targetY);
    wrong_count = (wrong_count + 1) % 3;
  }

  // Always return gantry system safely back to 0,0 before enabling conveyor movement
  moveToCoordinate(POS_0, POS_0);
}