# ---------------------------------------------------------------------------
# The tool schema is generated into a header from the canonical JSON file, so
# there is a single source of truth shared with the host tools.
# ---------------------------------------------------------------------------
set(ND_SCHEMA_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../tools/led.json")

file(READ "${ND_SCHEMA_PATH}" ND_SCHEMA_RAW)
string(REGEX REPLACE "[\r\n]" "" ND_SCHEMA_RAW "${ND_SCHEMA_RAW}")
string(STRIP "${ND_SCHEMA_RAW}" ND_SCHEMA_RAW)
string(REPLACE "\\" "\\\\" ND_SCHEMA_ESC "${ND_SCHEMA_RAW}")
string(REPLACE "\"" "\\\"" ND_SCHEMA_ESC "${ND_SCHEMA_ESC}")
configure_file("${CMAKE_CURRENT_LIST_DIR}/tools_schema.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/generated/tools_schema.h" @ONLY)

# Re-run CMake when the schema changes so the header never goes stale.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ND_SCHEMA_PATH}")

idf_component_register(SRCS "main.c"
                       INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/generated"
                       REQUIRES needle esp_driver_rmt esp_driver_gpio
                                esp_partition esp_timer)
