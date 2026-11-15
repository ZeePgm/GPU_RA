#pragma once
#include <string>
inline std::string plan_ptx(unsigned width,bool timed) {
 std::string s=R"(.version 9.0
.target sm_75
.address_size 64
.visible .entry r3_read(.param .u64 input,.param .u64 plan,.param .u64 output) {
.reg .b32 %r<16>;
.reg .b64 %rd<16>;
.reg .pred %p<2>;
ld.param.u64 %rd0,[input];
ld.param.u64 %rd1,[plan];
ld.param.u64 %rd2,[output];
mov.u32 %r0,%tid.x;
mov.u32 %r1,%ctaid.x;
mov.u32 %r2,%ntid.x;
mad.lo.u32 %r3,%r1,%r2,%r0;
mul.wide.u32 %rd3,%r3,4;
add.u64 %rd4,%rd1,%rd3;
ld.global.u32 %r4,[%rd4];
setp.ne.u32 %p0,%r4,4294967295;
cvt.u64.u32 %rd5,%r4;
add.u64 %rd6,%rd0,%rd5;
mul.wide.u32 %rd7,%r3,32;
add.u64 %rd8,%rd2,%rd7;
)";
 if(timed)s+="mov.u64 %rd10,%globaltimer;\n";
 s+=width==16?"@%p0 ld.global.v4.u32 {%r8,%r9,%r10,%r11},[%rd6];\n":"@%p0 ld.global.u32 %r8,[%rd6];\n";
 if(timed)s+="mov.u64 %rd11,%globaltimer;\n";
 s+=width==16?"@%p0 st.global.v4.u32 [%rd8],{%r8,%r9,%r10,%r11};\n":"@%p0 st.global.u32 [%rd8],%r8;\n";
 if(timed)s+="@%p0 st.global.u64 [%rd8+16],%rd10;\n@%p0 st.global.u64 [%rd8+24],%rd11;\n";
 return s+"ret;\n}\n";
}
