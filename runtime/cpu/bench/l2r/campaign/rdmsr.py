import os, struct, sys
cpu, reg = int(sys.argv[1]), int(sys.argv[2], 0)
fd = os.open(f"/dev/cpu/{cpu}/msr", os.O_RDONLY)
print(hex(struct.unpack("<Q", os.pread(fd, 8, reg))[0]))
