// === 电机控制主程序 ===

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "bdc_motor.h"
#include "driver/uart.h"

static const char *TAG = "app_main";

#define PWM_GPIO_M1A 5
#define PWM_GPIO_M1B 4
#define PWM_GPIO_M2A 16
#define PWM_GPIO_M2B 15
#define PWM_GPIO_M3A 10
#define PWM_GPIO_M3B 9
#define PWM_GPIO_M4A 14
#define PWM_GPIO_M4B 13

#define PWM_MOTOR_FREQ_HZ 1000
#define PWM_MOTOR_TIMER_RESOLUTION_HZ 100000

#define PWM_MOTOR_TIMER_GROUP_ID_M1 0
#define PWM_MOTOR_TIMER_GROUP_ID_M2 0
#define PWM_MOTOR_TIMER_GROUP_ID_M3 0
#define PWM_MOTOR_TIMER_GROUP_ID_M4 1

#define FIXED_SPEED 50
#define UART_NUM UART_NUM_0
#define BUF_SIZE 1024

bdc_motor_handle_t motor_m1, motor_m2, motor_m3, motor_m4;

void PwmMotor_Init()
{
    ESP_LOGI(TAG, "Initializing motors...");

    bdc_motor_config_t config;
    bdc_motor_mcpwm_config_t mcpwm;

    config.pwm_freq_hz = PWM_MOTOR_FREQ_HZ;
    mcpwm.resolution_hz = PWM_MOTOR_TIMER_RESOLUTION_HZ;

    config.pwma_gpio_num = PWM_GPIO_M1A;
    config.pwmb_gpio_num = PWM_GPIO_M1B;
    mcpwm.group_id = PWM_MOTOR_TIMER_GROUP_ID_M1;
    bdc_motor_new_mcpwm_device(&config, &mcpwm, &motor_m1);
    bdc_motor_enable(motor_m1);

    config.pwma_gpio_num = PWM_GPIO_M2A;
    config.pwmb_gpio_num = PWM_GPIO_M2B;
    mcpwm.group_id = PWM_MOTOR_TIMER_GROUP_ID_M2;
    bdc_motor_new_mcpwm_device(&config, &mcpwm, &motor_m2);
    bdc_motor_enable(motor_m2);

    config.pwma_gpio_num = PWM_GPIO_M3A;
    config.pwmb_gpio_num = PWM_GPIO_M3B;
    mcpwm.group_id = PWM_MOTOR_TIMER_GROUP_ID_M3;
    bdc_motor_new_mcpwm_device(&config, &mcpwm, &motor_m3);
    bdc_motor_enable(motor_m3);

    config.pwma_gpio_num = PWM_GPIO_M4A;
    config.pwmb_gpio_num = PWM_GPIO_M4B;
    mcpwm.group_id = PWM_MOTOR_TIMER_GROUP_ID_M4;
    bdc_motor_new_mcpwm_device(&config, &mcpwm, &motor_m4);
    bdc_motor_enable(motor_m4);

    ESP_LOGI(TAG, "Motors initialized");
}

void stop_all_motors()
{
    bdc_motor_brake(motor_m1);
    bdc_motor_brake(motor_m2);
    bdc_motor_brake(motor_m3);
    bdc_motor_brake(motor_m4);
}

// 前进
void forward_all_motors(int speed)
{
    bdc_motor_forward(motor_m1); // 左前
    bdc_motor_set_speed(motor_m1, speed);
    bdc_motor_forward(motor_m2); // 左后
    bdc_motor_set_speed(motor_m2, speed);

    bdc_motor_reverse(motor_m3); // 右前
    bdc_motor_set_speed(motor_m3, speed);
    bdc_motor_reverse(motor_m4); // 右后
    bdc_motor_set_speed(motor_m4, speed);
}

// 后退
void backward_all_motors(int speed)
{
    bdc_motor_reverse(motor_m1); // 左前
    bdc_motor_set_speed(motor_m1, speed);
    bdc_motor_reverse(motor_m2); // 左后
    bdc_motor_set_speed(motor_m2, speed);

    bdc_motor_forward(motor_m3); // 右前
    bdc_motor_set_speed(motor_m3, speed);
    bdc_motor_forward(motor_m4); // 右后
    bdc_motor_set_speed(motor_m4, speed);
}

// 原地左转
void turn_left()
{
    bdc_motor_reverse(motor_m1);
    bdc_motor_set_speed(motor_m1, FIXED_SPEED);
    bdc_motor_reverse(motor_m2);
    bdc_motor_set_speed(motor_m2, FIXED_SPEED);
    bdc_motor_reverse(motor_m3);
    bdc_motor_set_speed(motor_m3, FIXED_SPEED);
    bdc_motor_reverse(motor_m4);
    bdc_motor_set_speed(motor_m4, FIXED_SPEED);
}

// 原地右转
void turn_right()
{
    bdc_motor_forward(motor_m1);
    bdc_motor_set_speed(motor_m1, FIXED_SPEED);
    bdc_motor_forward(motor_m2);
    bdc_motor_set_speed(motor_m2, FIXED_SPEED);
    bdc_motor_forward(motor_m3);
    bdc_motor_set_speed(motor_m3, FIXED_SPEED);
    bdc_motor_forward(motor_m4);
    bdc_motor_set_speed(motor_m4, FIXED_SPEED);
}

void uart_control_task(void *arg)
{
    uint8_t data[BUF_SIZE] = {0};
    int index = 0;

    uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    };

    uart_driver_install(UART_NUM, BUF_SIZE * 2, 0, 0, NULL, 0);
    uart_param_config(UART_NUM, &uart_config);
    uart_set_pin(UART_NUM, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE,
                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);

    while (1)
    {
        int len = uart_read_bytes(UART_NUM, data + index, 1, pdMS_TO_TICKS(100));
        if (len > 0)
        {
            if (data[index] == '\n' || index >= BUF_SIZE - 1)
            {
                data[index] = '\0';
                ESP_LOGI(TAG, "收到命令: %s", data);

                // 只处理单字符指令
                if (strcmp((char *)data, "F") == 0) forward_all_motors(FIXED_SPEED);
                else if (strcmp((char *)data, "B") == 0) backward_all_motors(FIXED_SPEED);
                else if (strcmp((char *)data, "L") == 0) turn_left();
                else if (strcmp((char *)data, "R") == 0) turn_right();
                else if (strcmp((char *)data, "S") == 0) stop_all_motors();
                else stop_all_motors();

                index = 0;
                memset(data, 0, BUF_SIZE);
            }
            else
            {
                index++;
            }
        }
    }
}

void app_main(void)
{
    printf("Start Yahboom car\n");
    PwmMotor_Init();
    xTaskCreate(uart_control_task, "uart_control_task", 4096, NULL, 10, NULL);
    stop_all_motors();
}