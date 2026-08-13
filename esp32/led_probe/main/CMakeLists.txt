idf_component_register(SRCS "main.c"
                       REQUIRES driver esp_driver_rmt esp_driver_gpio
                       PRIV_REQUIRES esp_psram)
