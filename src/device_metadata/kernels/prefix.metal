#include <metal_stdlib>
using namespace metal;
typedef ulong wm_u64;
#define WM_DEVICE device
#define WM_PRIVATE thread
#define WM_CONSTANT constant
#define WM_FN inline
#define WM_LOG precise::log
#define WM_ATOMIC_OR(p,v) atomic_fetch_or_explicit((device atomic_uint*)(p),v,memory_order_relaxed)
#define WM_ATOMIC_EXCHANGE(p,v) atomic_exchange_explicit((device atomic_uint*)(p),v,memory_order_relaxed)
