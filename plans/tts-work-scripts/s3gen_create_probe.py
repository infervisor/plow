import ctypes, sys
lib = ctypes.CDLL("/root/tts-work/assets/cbx/codec/libplow_s3gen.so")
for mt in (600, 1000):
    h = ctypes.c_void_p()
    rc = lib.plow_s3gen_create(0, b"/root/tts-work/assets/cbx/codec/s3gen.bin", 1, mt, ctypes.byref(h))
    print("max_tokens", mt, "rc", rc)
    if rc == 0:
        lib.plow_s3gen_destroy(h)
