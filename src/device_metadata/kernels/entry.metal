kernel void watermark_init(device const uchar* seed [[buffer(0)]], device uchar* output [[buffer(1)]],
                            device uint* random [[buffer(2)]], constant uint* p [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  wm_initialize(seed,output,random,p[0],p[1],p[2],p[3],tid);
}
kernel void watermark_repair(device const uchar* seed [[buffer(0)]], device uchar* output [[buffer(1)]],
                            device uint* random [[buffer(2)]], constant uint* p [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  wm_repair(seed,random,p[0],p[1],tid);
}
kernel void watermark_links(device const uchar* seed [[buffer(0)]], device uchar* output [[buffer(1)]],
                            device uint* random [[buffer(2)]], constant uint* p [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  wm_links(random,p[1],tid);
}
kernel void watermark_parents(device const uchar* seed [[buffer(0)]], device uchar* output [[buffer(1)]],
                            device uint* random [[buffer(2)]], constant uint* p [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  wm_parents(random,p[1],tid);
}
kernel void watermark_resolve(device const uchar* seed [[buffer(0)]], device uchar* output [[buffer(1)]],
                            device uint* random [[buffer(2)]], constant uint* p [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  wm_resolve(output,random,p[1],tid);
}
kernel void watermark_input(device const uchar* seed [[buffer(0)]], device const uint* input [[buffer(1)]],
                            device uchar* output [[buffer(2)]], constant uint* p [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  wm_input(seed,input,output,p[0],p[1],p[2],p[3],p[4],p[5],tid);
}
