#include <stdio.h>
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

void app_main(void)
{
    /* The engine's three big PSRAM allocations, probed for real. */
    const size_t kv    = 27u * 256u * 256u * 2u;  /* k+v int8, 27 layers   */
    const size_t scale = 27u * 256u * 4u * 2u * sizeof(float);
    void *a = heap_caps_malloc(kv,    MALLOC_CAP_SPIRAM);
    void *b = heap_caps_malloc(scale, MALLOC_CAP_SPIRAM);

    for (int i = 0;; i++) {
        printf("HB %d psram_total=%u psram_free=%u internal_free=%u kv(%u)=%s scales(%u)=%s\n",
               i,
               (unsigned)heap_caps_get_total_size(MALLOC_CAP_SPIRAM),
               (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
               (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
               (unsigned)kv, a ? "OK" : "FAIL",
               (unsigned)scale, b ? "OK" : "FAIL");
        fflush(stdout);
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}
