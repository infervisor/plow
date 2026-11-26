"""mkrows.py <op_test.cu> <out.cu>: standalone FP8 decode GEMV driver from sm120_interp_op_test.cu
(its G7 fp8 GEMV section + main's fp8 GEMV calls; the full op test does not compile at HEAD:
Gemma MoE test call sites lag their bodies)."""
import sys
src = open(sys.argv[1]).read()
lines = src.split("\n")
ri = src.index("static void report(")
head = src[: src.index("\n}\n", ri) + 3]
a = src.index("/* ==================== fp8 (w8a16) DECODE GEMV (G7)")
b = src.index("if (dWu) cudaFree(dWu);\n}\n", a) + len("if (dWu) cudaFree(dWu);\n}\n")
calls = [l for l in lines if l.strip().startswith(("test_gemv_fp8(", "test_gemv_glu_fp8(", "test_gemv_fp8_rows("))]
main = "int main() {\n" + "\n".join(calls) + '\n    printf("fp8 gemv driver %s\\n", g_fail ? "FAIL" : "PASS");\n    return g_fail;\n}\n'
open(sys.argv[2], "w").write(head + "\n" + src[a:b] + "\n" + main)
print(len(calls), "calls")
