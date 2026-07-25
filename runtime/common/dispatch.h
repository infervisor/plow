/* dispatch.h — opcode → function-pointer dispatch table.
 *
 * The table is backend-specific (the loaded table already implies a backend),
 * so it is indexed by `family<<8 | variant` = the low 12 bits of the opcode.
 */
#ifndef PLOW_DISPATCH_H
#define PLOW_DISPATCH_H

#include <stdint.h>
#include "packet.h"
#include "kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PLOW_OP_SLOTS 4096 /* 16 families × 256 variants */

typedef struct dispatch_table {
    kernel_fn fn[PLOW_OP_SLOTS];
} dispatch_table;

/* family|variant index within a backend's table. */
static inline int plow_op_index(uint16_t opcode) { return opcode & 0x0FFF; }

/* Zero every slot. */
void dt_init(dispatch_table* dt);

/* Bind `fn` to an opcode's (family,variant) slot. */
void dt_register(dispatch_table* dt, uint16_t opcode, kernel_fn fn);

/* Look up the kernel for an opcode, or NULL if unregistered. */
kernel_fn dt_lookup(const dispatch_table* dt, uint16_t opcode);

/* Resolve+dispatch one instruction. Returns 0 on success, -1 if no kernel is
 * registered for the opcode. */
int dt_dispatch(const dispatch_table* dt, uint16_t opcode, const void* body, kctx* ctx);

#ifdef __cplusplus
}
#endif

#endif /* PLOW_DISPATCH_H */
