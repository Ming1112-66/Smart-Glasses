import processing.serial.*;

Serial myPort;
String val;
int[] valuesA;
int[] valuesB;

void setup() {
  size(900, 800);
  println(Serial.list());
  String portName = Serial.list()[0];
  
  try {
    myPort = new Serial(this, portName, 115200);
  } catch (Exception e) {
    println("Error opening port: " + e.getMessage());
    exit();
  }
  
  valuesA = new int[15];
  valuesB = new int[15];
}

void draw() {
  background(255);
  
  int w = width / 6;
  int h = height / 5;
  
  drawAndLabelRectangles(valuesA, "A", 3, w, h, false); // A区从左到右
  drawAndLabelRectangles(valuesB, "B", 0, w, h, true);  // B区从右到左
}

void drawAndLabelRectangles(int[] values, String label, int colOffset, int w, int h, boolean reverse) {
  for (int i = 0; i < values.length; i++) {
    fill(values[i]);
    int colIndex = i % 3;
    if (reverse) colIndex = 2 - colIndex; // 如果是反转的，调整列索引
    int x = (colIndex + colOffset) * w;
    int y = (i / 3) * h;
    rect(x, y, w, h);
    fill(values[i] < 128 ? 255 : 0);
    String text = label + (i + 1);
    textAlign(CENTER, CENTER);
    textSize(20);
    text(text, x + w / 2, y + h / 2);
  }
}

void serialEvent(Serial myPort) {
  val = myPort.readStringUntil('\n');
  if (val != null) {
    val = trim(val);
    println("Received: " + val);
    
    String[] parts = split(val, ',');
    if (parts.length == 31 && parts[0].equals("A")) { // 期望有31个部分
      for (int i = 1; i < parts.length/2+1; i += 1) {
        int index = i-1;
        if (index < valuesA.length && match(parts[i], "^[0-9]+$") != null) {
          valuesA[index] = int(map(int(parts[i]), 0, 100, 0, 255));
          println("Parsed for group A: index=" + index + " value=" + parts[i]);
        }
        if (index < valuesB.length && i + 14 < parts.length && match(parts[i + 14], "^[0-9]+$") != null) {
          valuesB[index] = int(map(int(parts[i + 14]), 0, 100, 0, 255));
          println("Parsed for group B: index=" + index + " value=" + parts[i+14]);
        }
      }
    } else {
      println("Error parsing: Invalid input " + val);
    }
  }
}
