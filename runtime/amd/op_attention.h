/* op_attention.h — FlashAttention family (dense GQA, MLA, DSA indexer): the ARCH SELECTOR.
 *
 * The family is three files, and this one only picks the arm:
 *
 *   op_attention_gfx942.h   CDNA3 (MI300X): arch-defaulted knobs and refusals
 *   op_attention_gfx950.h   CDNA4 (MI350X/MI355X): the same, plus the gfx950-only TR16 path
 *   op_attention_common.h   the shared bodies and dispatch glue, keyed ONLY on what the arch
 *                           file defined -- never on PLOW_CDNA4 itself
 *
 * An arch file is the tuning surface for its silicon: an experiment that must not move the other
 * arch's objects lives there (and is `#error`ed in the other arch file if it has no meaning
 * there). Every include site keeps including THIS file; the arm is chosen by PLOW_CDNA4 from
 * amd_arch.h, exactly the macro the old single header keyed its `#if` arms on.
 */
#ifndef PLOW_OP_ATTENTION_H
#define PLOW_OP_ATTENTION_H

#include "amd_common.h" /* amd_arch.h: PLOW_CDNA4 */

#if PLOW_CDNA4
#include "op_attention_gfx950.h"
#else
#include "op_attention_gfx942.h"
#endif

#endif /* PLOW_OP_ATTENTION_H */
