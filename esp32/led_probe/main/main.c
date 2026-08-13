/* led_probe - identify the onboard LED type and verify the external red LED.
 *
 * Runs three phases in a loop, announcing each over USB serial:
 *   A) GPIO4 (external red LED on the breadboard) - 3 slow blinks
 *   B) GPIO48 driven as an addressable WS2812   - red, green, blue
 *   C) GPIO48 driven as a plain GPIO            - 5 fast blinks
 *
 * Whichever of B or C actually lights the onboard LED tells us how to drive
 * it in the real firmware. Both are attempted every cycle; driving a plain
 * LED with WS2812 timings just produces a brief flicker, and driving a
 * WS2812 with a static level does nothing, so neither phase can damage the
 * board.
 */
#include <stdio.h>

#include "driver/gpio.h"
#include "driver/rmt_tx.h"
#include "esp_log.h"
#include "esp_check.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_heap_caps.h"
#include "esp_psram.h"

#define RED_LED_GPIO     4
#define ONBOARD_GPIO     48
#define RMT_RESOLUTION   10000000 /* 10 MHz -> 100 ns per tick */

static const char *TAG = "probe";

/* WS2812 bit timings at 100 ns per tick. */
static const rmt_symbol_word_t WS_BIT0 = {
    .level0 = 1, .duration0 = 3,   /* T0H 0.3 us */
    .level1 = 0, .duration1 = 9,   /* T0L 0.9 us */
};
static const rmt_symbol_word_t WS_BIT1 = {
    .level0 = 1, .duration0 = 9,   /* T1H 0.9 us */
    .level1 = 0, .duration1 = 3,   /* T1L 0.3 us */
};

static rmt_channel_handle_t s_chan;
static rmt_encoder_handle_t s_encoder;

static esp_err_t ws2812_init(void)
{
    rmt_tx_channel_config_t chan_cfg = {
        .clk_src           = RMT_CLK_SRC_DEFAULT,
        .gpio_num          = ONBOARD_GPIO,
        .mem_block_symbols = 64,
        .resolution_hz     = RMT_RESOLUTION,
        .trans_queue_depth = 4,
    };
    rmt_bytes_encoder_config_t enc_cfg = {
        .bit0 = WS_BIT0,
        .bit1 = WS_BIT1,
        .flags = { .msb_first = 1 },   /* WS2812 takes the MSB of each byte first */
    };

    ESP_RETURN_ON_ERROR(rmt_new_tx_channel(&chan_cfg, &s_chan), TAG, "tx channel");
    ESP_RETURN_ON_ERROR(rmt_new_bytes_encoder(&enc_cfg, &s_encoder), TAG, "encoder");
    ESP_RETURN_ON_ERROR(rmt_enable(s_chan), TAG, "enable");
    return ESP_OK;
}

static void ws2812_set(uint8_t r, uint8_t g, uint8_t b)
{
    /* WS2812 wants GRB order. */
    uint8_t grb[3] = { g, r, b };
    rmt_transmit_config_t tx = { .loop_count = 0 };

    ESP_ERROR_CHECK(rmt_transmit(s_chan, s_encoder, grb, sizeof(grb), &tx));
    ESP_ERROR_CHECK(rmt_tx_wait_all_done(s_chan, portMAX_DELAY));
}

static void ws2812_release(void)
{
    ESP_ERROR_CHECK(rmt_disable(s_chan));
    ESP_ERROR_CHECK(rmt_del_encoder(s_encoder));
    ESP_ERROR_CHECK(rmt_del_channel(s_chan));
    s_chan = NULL;
    s_encoder = NULL;
}

static void plain_gpio_blink(int gpio, int times, int on_ms, int off_ms)
{
    gpio_config_t cfg = {
        .pin_bit_mask = 1ULL << gpio,
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&cfg));

    for (int i = 0; i < times; i++) {
        gpio_set_level(gpio, 1);
        vTaskDelay(pdMS_TO_TICKS(on_ms));
        gpio_set_level(gpio, 0);
        vTaskDelay(pdMS_TO_TICKS(off_ms));
    }
}

void app_main(void)
{
    /* Give the USB serial console a moment to attach before the first log. */
    vTaskDelay(pdMS_TO_TICKS(1500));

    ESP_LOGI(TAG, "==================================================");
    ESP_LOGI(TAG, "Needle ESP32-S3 LED probe");
    ESP_LOGI(TAG, "external red LED on GPIO%d, onboard on GPIO%d",
             RED_LED_GPIO, ONBOARD_GPIO);
    ESP_LOGI(TAG, "==================================================");
#if CONFIG_SPIRAM
    ESP_LOGI(TAG, "PSRAM: %u bytes total, %u free",
             (unsigned)heap_caps_get_total_size(MALLOC_CAP_SPIRAM),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    {
        /* The KV cache is the single biggest allocation the engine makes. */
        const size_t kv = 27u * 256u * 256u * 2u;
        void *p = heap_caps_malloc(kv, MALLOC_CAP_SPIRAM);
        ESP_LOGI(TAG, "KV-cache probe (%u bytes): %s", (unsigned)kv,
                 p ? "OK" : "FAILED");
        free(p);
    }
#else
    ESP_LOGW(TAG, "PSRAM disabled in this build");
#endif
    ESP_LOGI(TAG, "internal free: %u", (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));

    for (int cycle = 1;; cycle++) {
        ESP_LOGI(TAG, "");
        ESP_LOGI(TAG, "--- cycle %d ---", cycle);

        ESP_LOGI(TAG, "[A] external red LED on GPIO%d: 3 slow blinks", RED_LED_GPIO);
        plain_gpio_blink(RED_LED_GPIO, 3, 400, 400);
        vTaskDelay(pdMS_TO_TICKS(800));

        ESP_LOGI(TAG, "[B] onboard as WS2812: RED, GREEN, BLUE (1s each)");
        if (ws2812_init() == ESP_OK) {
            ws2812_set(64, 0, 0);
            vTaskDelay(pdMS_TO_TICKS(1000));
            ws2812_set(0, 64, 0);
            vTaskDelay(pdMS_TO_TICKS(1000));
            ws2812_set(0, 0, 64);
            vTaskDelay(pdMS_TO_TICKS(1000));
            ws2812_set(0, 0, 0);
            ws2812_release();
        } else {
            ESP_LOGW(TAG, "    WS2812 init failed");
        }
        vTaskDelay(pdMS_TO_TICKS(800));

        ESP_LOGI(TAG, "[C] onboard as plain GPIO: 5 fast blinks");
        plain_gpio_blink(ONBOARD_GPIO, 5, 150, 150);

        ESP_LOGI(TAG, "--- end cycle %d, pausing 3s ---", cycle);
        vTaskDelay(pdMS_TO_TICKS(3000));
    }
}
