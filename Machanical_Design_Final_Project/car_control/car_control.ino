#include <PS2X_lib.h>  //for v1.6#include <PS2X_lib.h>  //for v1.6

PS2X ps2x; // 建立PS2控制器的類別實體
#include <Servo.h> //Reading servo motor library
Servo upservo; //Create an object for the servo
Servo upservo2;
Servo addservo;
Servo goservo; //Create an object for the servo
int val; //Variable for storing servo angle
int a=0;
int pos;
int angle;
int addpos;
/*
此程式庫不支援熱插拔，亦即，你必須在連接控制器後重新啟動Arduino，
或者在連接控制器之後再次呼叫config_gamepad()函數。
*/
/* define L298N motor drive control pins */
int RightMotorForward = 3;    // IN1
int RightMotorBackward = 5;   // IN2
int LeftMotorForward = 6;     // IN3
int LeftMotorBackward =11;    // IN4
const byte RIGHT_EN = 0;
const byte LEFT_EN = 2;
// Define variable to store last code received
unsigned long lastCode; /* define L298N motor drive control pins */
int error = 0; 
byte type = 0;
byte vibrate = 0;
 
void setup(){
 Serial.begin(57600);

 // ********注意！******** 1.6版的新語法：
 // 控制器接腳設置並驗證是否有錯誤:  GamePad(時脈腳位, 命令腳位, 選取腳位, 資料腳位, 是否支援類比按鍵, 是否支援震動) 
 error = ps2x.config_gamepad(13,A0,10,12, true, true);
 
 if(error == 0) { // 如果控制器連接沒有問題，就顯示底下的訊息。
   Serial.println("Found Controller, configured successful");
   Serial.println("Try out all the buttons, X will vibrate the controller, faster as you press harder;");
   Serial.println("holding L1 or R1 will print out the analog stick values.");
   Serial.println("Go to www.billporter.info for updates and to report bugs.");
 }
   
  else if(error == 1) // 找不到控制器，顯示底下的錯誤訊息。
   Serial.println("No controller found, check wiring, see readme.txt to enable debug. visit www.billporter.info for troubleshooting tips");
   
  else if(error == 2)  // 發現控制器，但不接受命令，請參閱程式作者網站的除錯說明。
   Serial.println("Controller found but not accepting commands. see readme.txt to enable debug. Visit www.billporter.info for troubleshooting tips");
   
  else if(error == 3) // 控制器拒絕進入類比感測壓力模式，或許是此控制器不支援的緣故。
   Serial.println("Controller refusing to enter Pressures mode, may not support it. ");
   
   type = ps2x.readType();  // 偵測控制器器的類型
     switch(type) {
       case 0:
        Serial.println("Unknown Controller type");   // 未知的控制器類型
       break;
       case 1:
        Serial.println("DualShock Controller Found");  // 發現DualShock控制器
       break;
       case 2:
         Serial.println("GuitarHero Controller Found");  // 發現吉他英雄控制器
       break;
     }

     /////////servoservo----------------------------------------------------------------------------------------------------
  upservo.attach(7); //Set digital pin 9 as the command pin for determining the servo angle
  upservo2.attach(4);
  addservo.attach(2);
  goservo.attach(8); //Set digital pin 9 as the command pin for determining the servo angle
  /* initialize motor control pins as output */
  pinMode(LeftMotorForward,OUTPUT);
  pinMode(RightMotorForward,OUTPUT);
  pinMode(LeftMotorBackward,OUTPUT);
  pinMode(RightMotorBackward,OUTPUT);
  pinMode(LEFT_EN, OUTPUT);  
  pinMode(RIGHT_EN, OUTPUT);
  pinMode(13,OUTPUT);
}

void loop(){
   /* 
   你必須執行ps2x.read_gamepad()方法來獲取新的按鍵值，語法格式：
   ps2x.read_gamepad(小馬達開或關, 大馬達強度值從0~255)
   如果不啟用震動功能，請執行
   ps2x.read_gamepad();
   不需要任何參數。
   
   你應該至少一秒鐘執行一次這個方法。
   */
 if(error == 1) // 如果沒發現任何控制器，則跳出迴圈。
  return; 
  
 if(type == 1) { // 這是標準的DualShock控制器
    ps2x.read_gamepad();      // 讀取控制器並且命令大的震動馬達以"vibrate"變數值的速度旋轉
    
    if(ps2x.Button(PSB_START))          // 查看「開始」鍵是否被按住
      Serial.println("Start is being held");
    if(ps2x.Button(PSB_SELECT))       // 查看「選擇」鍵是否被按住
      Serial.println("Select is being held");
         
    if(ps2x.Button(PSB_PAD_UP)) {         // 若「上」按鍵被按著
      Serial.print("Up held this hard: ");
      Serial.println(ps2x.Analog(PSAB_PAD_UP), DEC);
      MotorFoward();
    }
    if(ps2x.Button(PSB_PAD_RIGHT)){
      Serial.print("Right held this hard: ");
      Serial.println(ps2x.Analog(PSAB_PAD_RIGHT), DEC);
      TurnRight();
    }
    if(ps2x.Button(PSB_PAD_LEFT)){
      Serial.print("LEFT held this hard: ");
      Serial.println(ps2x.Analog(PSAB_PAD_LEFT), DEC);
      TurnLeft();
    }
    if(ps2x.Button(PSB_PAD_DOWN)){
      Serial.print("DOWN held this hard: ");
      Serial.println(ps2x.Analog(PSAB_PAD_DOWN), DEC);
      MotorBackward();
    }   
  
   vibrate = ps2x.Analog(PSAB_BLUE); // 依據你按著X按鍵的力道來調整馬達的震動強度 

    if (ps2x.NewButtonState())          // 若「按下」或「放開」任何按鈕
    {
        if(ps2x.Button(PSB_L2)){
         Serial.println("L2 pressed");
         back();//攀繩伺服馬達後退
        }
        if(ps2x.Button(PSB_R2)){
         Serial.println("R2 pressed");
         gogo();//攀繩伺服馬達前進
        }
        if(ps2x.ButtonPressed(PSB_RED)){            // 若「按下」圈圈按鍵
          Serial.println("Circle just pressed");
          pos=pos+10; //每按一下Circle鍵伺服馬達位置旋轉10度(下降)
          addpos=addpos-10;
          if(addpos<0) addpos=0;
          if(pos>180) pos=180;
          Serial.println("pos=");
          Serial.println(pos);
          Serial.println(addpos);
          writepos();
        }
        if(ps2x.ButtonPressed(PSB_PINK)){          // 若「放開」方塊按鍵
            Serial.println("Square just released"); 
            pos=pos-10;//每按一下Square鍵伺服馬達位置旋轉10度(上升)
            addpos=addpos+10;
            if(addpos>180) addpos=180;
            if(pos<0) pos=0;
            Serial.println("pos=");
            Serial.println(pos);
            writepos(); 
        }
    } 
    if(ps2x.Button(PSB_GREEN)) {   // 若被按下的是三角按鍵
         Serial.println("Triangle pressed"); //每按一下Square鍵伺服馬達位置歸零
         upup();
    }
    if(ps2x.Button(PSB_L1) || ps2x.Button(PSB_R1)){// 若被按下的是L1或是R1鍵，所有馬達暫停
      stopgogo();
      MotorStop();
    }

    
 }
 delay(50);
}

/* FORWARD */
void MotorFoward(){
  analogWrite(LeftMotorForward,165);//調低左側直流馬達轉速
  analogWrite(RightMotorForward,240);//升高右側直流馬達轉速
  digitalWrite(LeftMotorBackward,LOW);
  digitalWrite(RightMotorBackward,LOW); 

}
/* BACKWARD */
void MotorBackward(){
  analogWrite(LeftMotorBackward,180);
  analogWrite(RightMotorBackward,180);
  digitalWrite(LeftMotorForward,LOW);
  digitalWrite(RightMotorForward,LOW);
}

/* TURN RIGHT */
void TurnRight(){
  analogWrite(LeftMotorForward,180);
  digitalWrite(RightMotorForward,LOW);
  digitalWrite(LeftMotorBackward,LOW);
  analogWrite(RightMotorBackward,180);
}

/* TURN LEFT */
void TurnLeft(){
  analogWrite(RightMotorForward,180);
  digitalWrite(LeftMotorForward,LOW);
  analogWrite(LeftMotorBackward,180);
  digitalWrite(RightMotorBackward,LOW);
}

/* STOP */
void MotorStop(){
  digitalWrite(LeftMotorBackward,LOW);
  digitalWrite(RightMotorBackward,LOW);
  digitalWrite(LeftMotorForward,LOW);
  digitalWrite(RightMotorForward,LOW);
}
/*upup*/
void upup(){
  upservo.write(180);
  upservo2.write(0);
  pos=180;
  addpos=0;
  addservo.write(0);
  delay(15);  
}
 void writepos(){
  upservo.write(pos);
  upservo2.write(180-pos);
  addservo.write(addpos);
 }
/*down*/
void down(){
  upservo.write(110);
  upservo2.write(180-pos);
  pos=110;
  
}
/*GOGO*/
void gogo(){
  goservo.write(0);
}
/*Back*/
void back(){
    for (angle = 90; angle <= 180; angle += 1) { 
    goservo.write(angle);             
    delay(15); 
  goservo.write(angle);
   }//逐步增加馬達反轉轉速

}
/*stopgogo*/
void stopgogo(){
   goservo.write(90);
}
