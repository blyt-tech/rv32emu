/*
 * rv32emu is freely redistributable under the MIT License. See the file
 * "LICENSE" for information on usage and redistribution of this file.
 */

#pragma once

#include <stdint.h>

#include "riscv.h"
#include "riscv_private.h"

/* clang-format off */
enum {
    //                    ....xxxx....xxxx....xxxx....xxxx
    FMASK_SIGN        = 0b10000000000000000000000000000000,
    FMASK_EXPN        = 0b01111111100000000000000000000000,
    FMASK_FRAC        = 0b00000000011111111111111111111111,
    FMASK_QNAN        = 0b00000000010000000000000000000000,
    //                    ....xxxx....xxxx....xxxx....xxxx
    FFLAG_MASK        = 0b00000000000000000000000000011111,
    FFLAG_INVALID_OP  = 0b00000000000000000000000000010000,
    FFLAG_DIV_BY_ZERO = 0b00000000000000000000000000001000,
    FFLAG_OVERFLOW    = 0b00000000000000000000000000000100,
    FFLAG_UNDERFLOW   = 0b00000000000000000000000000000010,
    FFLAG_INEXACT     = 0b00000000000000000000000000000001,
    //                    ....xxxx....xxxx....xxxx....xxxx
    RV_NAN            = 0b01111111110000000000000000000000
};
/* clang-format on */

/* compute the fclass result */
static inline uint32_t calc_fclass(uint32_t f)
{
    const uint32_t sign = f & FMASK_SIGN;
    const uint32_t expn = f & FMASK_EXPN;
    const uint32_t frac = f & FMASK_FRAC;

    uint32_t out = 0;

    /*
     * 0x001    rs1 is -INF
     * 0x002    rs1 is negative normal
     * 0x004    rs1 is negative subnormal
     * 0x008    rs1 is -0
     * 0x010    rs1 is +0
     * 0x020    rs1 is positive subnormal
     * 0x040    rs1 is positive normal
     * 0x080    rs1 is +INF
     * 0x100    rs1 is a signaling NaN
     * 0x200    rs1 is a quiet NaN
     */

    /* Check the exponent bits */
    if (expn) {
        if (expn != FMASK_EXPN) {
            /* Check if it is negative normal or positive normal */
            out = sign ? 0x002 : 0x040;
        } else {
            /* Check if it is NaN */
            if (frac) {
                out = frac & FMASK_QNAN ? 0x200 : 0x100;
            } else if (!sign) {
                /* Check if it is +INF */
                out = 0x080;
            } else {
                /* Check if it is -INF */
                out = 0x001;
            }
        }
    } else if (frac) {
        /* Check if it is negative or positive subnormal */
        out = sign ? 0x004 : 0x020;
    } else {
        /* Check if it is +0 or -0 */
        out = sign ? 0x008 : 0x010;
    }

    return out;
}

static inline bool is_nan(uint32_t f)
{
    const uint32_t expn = f & FMASK_EXPN;
    const uint32_t frac = f & FMASK_FRAC;
    return (expn == FMASK_EXPN && frac);
}

/* Double-precision (RV32D) field masks and helpers — 64-bit siblings of the
 * single-precision constants above (Spike U). */
/* clang-format off */
enum {
    FMASK_SIGN_D = 0x8000000000000000ULL, /* bit 63          */
    FMASK_EXPN_D = 0x7FF0000000000000ULL, /* bits 62..52     */
    FMASK_FRAC_D = 0x000FFFFFFFFFFFFFULL, /* bits 51..0      */
    FMASK_QNAN_D = 0x0008000000000000ULL, /* bit 51          */
    RV_NAN_D     = 0x7FF8000000000000ULL  /* canonical qNaN  */
};
/* clang-format on */

/* compute the fclass result for a double */
static inline uint32_t calc_fclass_d(uint64_t f)
{
    const uint64_t sign = f & FMASK_SIGN_D;
    const uint64_t expn = f & FMASK_EXPN_D;
    const uint64_t frac = f & FMASK_FRAC_D;

    uint32_t out = 0;

    /* Same 10-way classification as calc_fclass (see above) */
    if (expn) {
        if (expn != FMASK_EXPN_D) {
            out = sign ? 0x002 : 0x040;
        } else {
            if (frac) {
                out = frac & FMASK_QNAN_D ? 0x200 : 0x100;
            } else if (!sign) {
                out = 0x080;
            } else {
                out = 0x001;
            }
        }
    } else if (frac) {
        out = sign ? 0x004 : 0x020;
    } else {
        out = sign ? 0x008 : 0x010;
    }

    return out;
}

static inline bool is_nan_d(uint64_t f)
{
    const uint64_t expn = f & FMASK_EXPN_D;
    const uint64_t frac = f & FMASK_FRAC_D;
    return (expn == FMASK_EXPN_D && frac);
}

/* ---- NaN-boxed FLEN=64 register-file accessors (Spike U) -----------------
 * The FP register file (rv->F[]) is 64-bit. A single-precision value is stored
 * NaN-boxed: low 32 bits hold the f32, upper 32 bits are all ones. Per the
 * RISC-V spec, a single-precision op that reads an improperly boxed source sees
 * the canonical NaN (0x7FC00000); single-precision results are written boxed.
 * Double-precision ops use the full 64 bits.
 */
#define FBOX_HI_MASK 0xFFFFFFFF00000000ULL
#define RV_CANON_NAN_F 0x7FC00000U

/* Read a single-precision operand with NaN-box validation. */
static inline softfloat_float32_t get_f32(const riscv_t *rv, uint8_t n)
{
    const uint64_t bits = rv->F[n].v;
    if ((bits & FBOX_HI_MASK) == FBOX_HI_MASK)
        return (softfloat_float32_t){.v = (uint32_t) bits};
    return (softfloat_float32_t){.v = RV_CANON_NAN_F};
}

/* Write a single-precision result, NaN-boxing into the high bits. */
static inline void set_f32(riscv_t *rv, uint8_t n, softfloat_float32_t v)
{
    rv->F[n].v = FBOX_HI_MASK | (uint64_t) v.v;
}

/* Raw low-32 read with no box check — for FSW and FMV.X.W, which transfer
 * bits [31:0] of the register verbatim. */
static inline uint32_t get_f32_bits(const riscv_t *rv, uint8_t n)
{
    return (uint32_t) rv->F[n].v;
}

static inline softfloat_float64_t get_f64(const riscv_t *rv, uint8_t n)
{
    return rv->F[n];
}

static inline void set_f64(riscv_t *rv, uint8_t n, softfloat_float64_t v)
{
    rv->F[n] = v;
}

static inline void set_fflag(riscv_t *rv)
{
    if (softfloat_exceptionFlags & softfloat_flag_invalid)
        rv->csr_fcsr |= FFLAG_INVALID_OP;
    if (softfloat_exceptionFlags & softfloat_flag_infinite)
        rv->csr_fcsr |= FFLAG_DIV_BY_ZERO;
    if (softfloat_exceptionFlags & softfloat_flag_overflow)
        rv->csr_fcsr |= FFLAG_OVERFLOW;
    if (softfloat_exceptionFlags & softfloat_flag_underflow)
        rv->csr_fcsr |= FFLAG_UNDERFLOW;
    if (softfloat_exceptionFlags & softfloat_flag_inexact)
        rv->csr_fcsr |= FFLAG_INEXACT;
    softfloat_exceptionFlags = 0;
}

static inline void set_rounding_mode(riscv_t *rv, uint8_t rm)
{
    const uint32_t frm = (rv->csr_fcsr >> 5) & (~(1 << 3));

    if (likely(rm == 0b111))
        rm = frm;

    switch (rm) {
    case 0b000:
        softfloat_roundingMode = softfloat_round_near_even;
        break;
    case 0b001:
        softfloat_roundingMode = softfloat_round_minMag;
        break;
    case 0b010:
        softfloat_roundingMode = softfloat_round_min;
        break;
    case 0b011:
        softfloat_roundingMode = softfloat_round_max;
        break;
    case 0b100:
        softfloat_roundingMode = softfloat_round_near_maxMag;
        break;
    default:
        __UNREACHABLE;
        break;
    }
}
