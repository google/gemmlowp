config_setting(
    name = "android",
    values = {
        "crosstool_top": "//external:android/crosstool",
    },
)

# Android builds do not need to link in a separate pthread library.
LIB_LINKOPTS = select({
    ":android": [],
    "//conditions:default": ["-lpthread"],
})

BIN_LINKOPTS = select({
    ":android": [],
    "//conditions:default": ["-lpthread"],
})
