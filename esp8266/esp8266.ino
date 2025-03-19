 #include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

const char* ssid = "duy";   // Thay bằng tên WiFi của bạn
const char* password = "11111111"; // Thay bằng mật khẩu WiFi

ESP8266WebServer server(80);

#define BUZZER_PIN 5  // GPIO5 (D1 trên NodeMCU)

void handleBuzzerOn() {
    digitalWrite(BUZZER_PIN, HIGH);
    server.send(200, "text/plain", "Buzzer ON");
}

void handleBuzzerOff() {
    digitalWrite(BUZZER_PIN, LOW);
    server.send(200, "text/plain", "Buzzer OFF");
}

void setup() {
    Serial.begin(115200);
    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);

    WiFi.begin(ssid, password);
    Serial.print("Đang kết nối WiFi...");
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.print(".");
    }
    Serial.println("\nWiFi kết nối thành công!");
    Serial.println(WiFi.localIP());

    server.on("/buzzer/on", handleBuzzerOn);
    server.on("/buzzer/off", handleBuzzerOff);

    server.begin();
    Serial.println("Server đã khởi động!");
}

void loop() {
    server.handleClient();
}
