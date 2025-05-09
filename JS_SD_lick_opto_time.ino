/*
 * Change the desired task variables and upload the sketch without micro SD card loaded. 
 * The sketch will start acquiring data upon uploading the sketch if the micro SD card is present.
 * 
 * The sketch will start up in the “SavingBroken” function and will not start until the Arduino is reset with a micro SD card loaded
 * 
 * Load the mouse and the micro SD card and press the onboard reset button to start the task
 * 
 * micro SD card should have only one file titled "Log.txt"
 * after collecting data run SavemicroSDData.m on MATLAB
 * 
 * MAN
 2023_06_19_v6_cleaned+TTL for optogenetics and camera
****************************************************************************************************/

 

/**************************************************************************************************
*
* Libraries
*
***************************************************************************************************/
#include <Wire.h>
#include <LiquidCrystal_I2C.h> //https://github.com/fdebrabander/Arduino-LiquidCrystal-I2C-library    
#include "Adafruit_MPR121.h"    //https://github.com/adafruit/Adafruit_MPR121     
#include <SD.h>
#include <SPI.h>


/**************************************************************************************************
*
*Config variables (do not modify)
*
***************************************************************************************************/
#ifndef _BV
#define _BV(bit) (1 << (bit)) //for MPR121 touch sensor
#endif

#define R A10 //random seed

#define LR_pin A0 //anolog pin for X // wg specyfikacji joysticka kable są wpięte na odwrót -> Y
#define UD_pin A1 //anolog pin for Y // wg specyfikacji joysticka kable są wpięte na odwrót -> X


File myFile; //for data saving 
File LOG; //initialize SD
String tlt; //title for CSV
int ct = 0; //for LOG file
int Andy; //random number added to CSV title 

Adafruit_MPR121 cap = Adafruit_MPR121();  // creat object for toch sensor
uint16_t lasttouched = 0;
uint16_t currtouched = 0;
bool lick_state = false;
bool opto_state = false;

/**************************************************************************************************
*
*Configuration of pins and adressess
*
***************************************************************************************************/
LiquidCrystal_I2C lcd(0x27, 16, 2);//0x27  0x3F
const int IRled_1 = 30; //pin out to IR LED1
const int IRled_2 = 32; //pin out to IR LED2
const int pinSD = 53; //pin out to SD card
const int mpr121_pin = 11; //pin on MPR121 out to lickport
const int Water_solenoid = 13; //pin out to water delivery valve
const int TTL_1 = 2; //pin out to camera
const int TTL_2 = 4; // pin out to RWD 
const int TTL_OPTO = 6; //pin out to Plexon LED driver
/**************************************************************************************************
Session variables (can be modified)
***************************************************************************************************/
unsigned long TTL_duration = 1;
unsigned long stim_duration = 300000; // opto stim duration in ms, costant light  
unsigned long base_duration = 360000; // base time in ms before opto stim starts - trials with no stim 
unsigned long stim_off = stim_duration + base_duration; 
unsigned long ms = 0; //stores time
unsigned long SolOpenTime = 0; //stores solenoid opening time
unsigned long SolCloseTime = 0; //stores solenoid closing time
unsigned long SolOpenDuration = 50; //solenoid duration time in ms
unsigned long thresholdcrossTime = 0; //stores threshold crossing time
unsigned long threshold = 90; //sets the threshold 10 points is ~1mm
//unsigned long thres_pull = 12; //sets the direction - pull
//unsigned long thres_push = -12; //sets the direction - push
unsigned long treshold_fail = 20 ; //
unsigned long DelayToRew = 1000; //pause till reward
unsigned long ITI = 3000; // time interval (in ms) betwen trials

int EM = 0; //first case in the main program (trial settings)
int NormalTrialCt = 0; //number of pre-stim trials
int StimTrialCt = 0; //number of stim trials in total
int PostStimTrialCt = 0; // number of post-stim trials
int TrialCt = 0; //number of trials in total
int Y = 0; //first joystick position
int X = 0; //first joystick position
int baseY = 0; //first joystick position
int baseX = 0; //first joystick position
int fail_attempts = 0; //number of failed attempts in a given trial
int sum_of_fail_attempts = 0; //number of failed attempts in total
int stim = 0;
int licks = 0; //number of licks in a given trial
int sum_licks = 0; //number of licks in total
int FrameTrigg_length = 10; //not sure?
String type_move;
float pos;



void setup() {
  //pin setup
  pinMode(IRled_1, OUTPUT); // sets the digital pin as output
  pinMode(IRled_2, OUTPUT); // sets the digital pin as output
  pinMode(Water_solenoid, OUTPUT); //
  pinMode(LED_BUILTIN, OUTPUT); //same as Water_solenoid PIN - generates blinks (solenoid opening) while loading the program
  pinMode(TTL_1, OUTPUT); // RWD threshold crossed
  pinMode(TTL_2, OUTPUT); // RWD opto stim period
  pinMode(TTL_OPTO, OUTPUT); //  drives optogenetic stimulation via PLEXON LED 
  pinMode(pinSD, OUTPUT);

  //set corresponding baud rate in ardunio serial monitor
  Serial.begin(115200); //set corresponding baud rate in Ardunio Serial Monitor
  
  //set base joystic position
  baseX = analogRead(LR_pin);
  baseY = analogRead(UD_pin);
  //turn on LEDs
  digitalWrite(IRled_1, HIGH); 
  digitalWrite(IRled_2, HIGH);

  //set camera recording  
  //digitalWrite(TTL_1, HIGH);
  
  //set address for MPR121: default is 0x5A, if tied to 3.3V its 0x5B, if tied to SDA its 0x5C and if SCL then 0x5D
  if (!cap.begin(0x5A)) {
    //Serial.println("MPR121 not found, check wiring?");
    while (1);
  }
    //Serial.println("MPR121 found!");
  
  //set LCD
  lcd.backlight();
  lcd.begin();

 //not sure exactly
  randomSeed(analogRead(R));
 
 //random file name  
  Andy = random(5000, 10001);
  tlt = String(Andy) + "_JS2.CSV"; //get random file name
  int ctt;
  File root;

  //SD config
  if (SD.begin()) { //initialize SD
    root = SD.open("/");
    ctt = printDirectory(root, 0); ctt = ctt - 2;
    Serial.println(String(ctt));
    if (ctt > 1) {
      TooManyFiles();
      return;
    }
    LOG = SD.open("Log.txt", FILE_READ); //SD card sgould have one file save titled "Log.txt"
    if (LOG) {
      while (LOG.available()) {
        Serial.println(LOG.readStringUntil('\n') + '_' + String(ct));
        ct++;
      }
      LOG.close();
    }
    else {
      SavingBroken();
      return;
    }
    LOG = SD.open("Log.txt", FILE_WRITE); //SD card sgould have one file save titled "Log.txt"
    if (LOG) {
      LOG.seek(LOG.position() + 1);
      LOG.println(String(ct) + ' ' + String(tlt));
      LOG.close();
    }
  }
  else
  {
    SavingBroken();
    return;
  }
    
}


void loop() {
  
  lcd.clear();
  Y = analogRead(UD_pin) - baseY;  X = analogRead(LR_pin) - baseX;
  pos = sqrt(pow(X, 2) + pow(Y, 2));
  Y = analogRead(UD_pin);  X = analogRead(LR_pin);

  
/*  
  //checks movement type - not sure if it works correctly
  type_move = "None" ;
  if (Y < thres_push) {
    type_move = "Push";
  }
  else if (Y > thres_pull){
    
    type_move = "Pull";
  }
  else {
     type_move = "None";
  }
*/
  

  //currently touched pads & lick detection
  currtouched = cap.touched();
  check_lick();
  check_opto();

/**************************************************************************************************
*
*Main program logic
*  
  Eventmarkers (EM)
  0 = not in trial
  1 = threshold met - delay
  2 = sol open
  3 = iti start
  4 = unrewarded movment (threshold not crossed)
***************************************************************************************************/
  
  switch (EM) {
    case 0: // read joystick position
      if ( treshold_fail < pos && pos < threshold) {   // GD: inicjacja ruchu przejdź do case 4 
         EM = 4;        
      }
      else if (pos > threshold) {
        thresholdcrossTime = millis(); 
        EM = 1;
        digitalWrite(TTL_1, HIGH); //threshold crossed 
        delay (TTL_duration);
        digitalWrite (TTL_1, LOW);
      }
      else {
        EM = 0;
      }
      break;

    case 1: // 1delay 2deliverWater 3iti
      if (DelayToRew <= millis() - thresholdcrossTime) {
        SolOpenTime = millis(); digitalWrite(Water_solenoid, HIGH); digitalWrite(LED_BUILTIN, HIGH);
       // digitalWrite(TTL_2, HIGH); //reward delivery
       // delay (TTL_duration);
       // digitalWrite (TTL_2, LOW);
        EM = 2;
      }
      else {
        EM = 1;
      }
      break;

    case 2:
      if (SolOpenDuration <= millis() - SolOpenTime) {
        SolCloseTime = millis(); digitalWrite(Water_solenoid, LOW); digitalWrite(LED_BUILTIN, LOW);
        EM = 3;
      }
      else {
        EM = 2;
      }
      break;

    case 3:
      if  (ITI <= millis() - SolCloseTime) { 
        baseY = analogRead(UD_pin); baseX = analogRead(LR_pin); 
        TrialCt++;
        sum_of_fail_attempts = sum_of_fail_attempts + fail_attempts;  
        sum_licks = sum_licks + licks;
        licks = 0;
        fail_attempts = 0;
        EM = 0;
        if ((opto_state == false) && (millis() < stim_off)) {
          NormalTrialCt++;
          }
        else if (opto_state == true) {
          StimTrialCt++;
        }
        else if ((opto_state == false) && (millis() > stim_off)) {
          PostStimTrialCt++;
        }

      }
      else {
        EM = 3;
      }
      break;
    
    case 4:
      if (pos > threshold) {                                      // GD: jeśli ruch joysticka się przesunie poza główny treshold idziemy do case 1 i noramlna kasakada 
        thresholdcrossTime = millis(); 
        EM = 1;
        digitalWrite(TTL_1, HIGH); //threshold crossed 
        delay (TTL_duration);
        digitalWrite (TTL_1, LOW);
        }
      else if  (pos < treshold_fail) {                            // GD: jeśli ruch joysticka się przesunie poniżej thresholdu dla inicacnji ruch to wracamy do case 0 i + 1 dla ruchów nienagrodzonych 
        fail_attempts++; EM = 0;
      }
      else {
        EM = 4;}
      break;
  }
  
  //data display
  lcd.setCursor (0, 0);
  lcd.print(String(TrialCt));
  lcd.setCursor (4, 0);
  lcd.print(String(NormalTrialCt));
  lcd.setCursor (8, 0);
  lcd.print(String(StimTrialCt));
  lcd.setCursor (12, 0);
  lcd.print(String(pos));
  lcd.setCursor (0, 1);
  lcd.print(String((ms / 1000) / 60));
  lcd.setCursor (5, 1);
  lcd.print(String(licks));
  lcd.setCursor (11, 1);
  lcd.print(String(threshold));




  myFile = SD.open(tlt, FILE_WRITE);
  if (myFile) {
    myFile.println(String(ms) + ',' +  String(EM) + ',' +  String(TrialCt) + ','  + String(X) + ',' + String(Y) + ',' + String(pos) + ',' + String(baseX) + ',' +  String(baseY) + ',' + String(SolOpenDuration) + ',' + String(DelayToRew) + ',' + String(ITI)
     + ',' + String(threshold) + ',' + String(fail_attempts) + ',' + String(sum_of_fail_attempts) + ',' + String(lick_state) + ',' + String(sum_licks) +',' + String(stim) + ',' + String(NormalTrialCt) + ',' + String(StimTrialCt) + ',' + String(PostStimTrialCt) + ',' + String(opto_state));
    myFile.close(); 
  }
  else {
    SavingBroken();
  }
 
}

// Opto time - controler

void check_opto(){
  ms = millis();
  if (( ms >  base_duration && ms < stim_off) && (opto_state == false))  {
    digitalWrite(TTL_OPTO, HIGH);
    digitalWrite(TTL_2, HIGH);
    opto_state = true;
  }
  else if ((ms > stim_off) && (opto_state == true)) {
    digitalWrite(TTL_OPTO, LOW);
    digitalWrite(TTL_2, LOW);
    opto_state = false;
  }

}


//lickometer state check
void check_lick(){
  //if it *is* touched and *wasnt* touched before, alert!
  if ((currtouched & _BV(mpr121_pin)) && !(lasttouched & _BV(mpr121_pin)) ) {
      lick_state = true ; licks++; //przeniesione z dołu, tu chyba lepiej pasuje?
    }
  // if it *was* touched and now *isnt*, alert!
  if (!(currtouched & _BV(mpr121_pin)) && (lasttouched & _BV(mpr121_pin)) ) {
      lick_state = false ; 
      
    }
    //resets our state
    lasttouched = currtouched;
}

//saving broken
void SavingBroken() {
  lcd.clear();
  lcd.print("If you did not");
  lcd.setCursor(0, 1);
  lcd.print("remove SD Card..");
  delay(3000);
  lcd.clear();
  lcd.print("Something is");
  lcd.setCursor(0, 1);
  lcd.print("broken");
  delay(3000);
  lcd.clear();
  lcd.print("Check the ");
  lcd.setCursor(0, 1);
  lcd.print("file...");
  delay(3000);
  lcd.clear();
  SavingBroken();
}

//SD saving
int printDirectory(File dir, int numTabs) {
  int ctt = 0;
  while (true) {
    File entry =  dir.openNextFile();
    ctt++;
    if (! entry) {
      return ctt;
      // no more files
      // return to the first file in the directory
      dir.rewindDirectory();
      break;
    }
    for (uint8_t i = 0; i < numTabs; i++) {
      Serial.print('\t');
    }
  }
}

//too many files
void TooManyFiles() {
  lcd.clear();
  lcd.print("There are too");
  lcd.setCursor(0, 1);
  lcd.print("many files");
  delay(3000);
  lcd.clear();  
}
