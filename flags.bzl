# Android builds do not need to link in a separate pthread library.
LIB_LINKOPTS = select({
    ":android": [],
    "//conditions:default": ["-lpthread"],
})

BIN_LINKOPTS = select({
    ":android": [],
    "//conditions:default": ["-lpthread"],
})
