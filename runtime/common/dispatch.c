#include "dispatch.h"
#include <string.h>

void dt_init(dispatch_table* dt) {
    memset(dt->fn, 0, sizeof(dt->fn));
}

void dt_register(dispatch_table* dt, uint16_t opcode, kernel_fn fn) {
    dt->fn[plow_op_index(opcode)] = fn;
}

kernel_fn dt_lookup(const dispatch_table* dt, uint16_t opcode) {
    return dt->fn[plow_op_index(opcode)];
}

int dt_dispatch(const dispatch_table* dt, uint16_t opcode, const void* body, kctx* ctx) {
    kernel_fn fn = dt->fn[plow_op_index(opcode)];
    if (!fn) return -1;
    fn(body, ctx);
    return 0;
}
