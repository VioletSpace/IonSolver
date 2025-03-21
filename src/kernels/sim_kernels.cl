//THESE ARE TO BE REMOVED
#if defined(FP16S) || defined(FP16C)
#define fpxx ushort
#else // FP32
#define fpxx float
#endif // FP32

#define DEF_NX 20u
#define DEF_NY 20u
#define DEF_NZ 20u
#define DEF_N 8000ul

#define DEF_DX 1u
#define DEF_DY 1u
#define DEF_DZ 1u
#define DEF_DI 0u

#define DEF_AX 1u
#define DEF_AY 1u
#define DEF_AZ 1u

#define DEF_OX 1u
#define DEF_OY 1u
#define DEF_OZ 1u

#define D "D2Q9" // D2Q9/D3Q15/D3Q19/D3Q27
#define DEF_VELOCITY_SET 9u // LBM velocity set (D2Q9/D3Q15/D3Q19/D3Q27)
#define DEF_DIMENSIONS 2u // number spatial dimensions (2D or 3D)
#define DEF_TRANSFERS 3u // number of DDFs that are transferred between multiple domains

#define DEF_C 0.57735027f // lattice speed of sound c = 1/sqrt(3)*dt
#define DEF_W 2.0f // relaxation rate w = dt/tau = dt/(nu/c^2+dt/2) = 1/(3*nu+1/2)
#define DEF_W0 (1.0f/2.25f)
#define DEF_WS (1.0f/9.0f)
#define DEF_WE (1.0f/36.0f)
#define DEF_KE 8.9875517923E9f
#define DEF_KMU 0.0f
#define def_ind_r 5 // Range of induction fill around cell
#define DEF_KMU0 0.0f
#define DEF_WQ 0.1f
#define DEF_KKGE 1.0f // Constant mass/charge for an electron
#define DEF_KIMG 1.0f // Inverse of mass of a propellant gas atom, scaled by 10^20
#define DEF_KVEV 1.0f // 9.10938356e-31kg / (2*1.6021766208e-19)

#define TYPE_S  0x01 // 0b00000001 // (stationary or moving) solid boundary
#define TYPE_E  0x02 // 0b00000010 // equilibrium boundary (inflow/outflow)
#define TYPE_C  0x04 // 0b00000100 // changing electric field
#define TYPE_F  0x08 // 0b00001000 // charged solid
#define TYPE_M  0x10 // 0b00010000 // magnetic solid
#define TYPE_G  0x20 // 0b00100000 // reserved type 3
#define TYPE_X  0x40 // 0b01000000 // reserved type 4
#define TYPE_Y  0x80 // 0b10000000 // reserved type 5
#define TYPE_MS 0x03 // 0b00000011 // cell next to moving solid boundary
#define TYPE_BO 0x03 // 0b00000011 // any flag bit used for boundaries

#define fpxx_copy ushort
#define load(p,o) half_to_float_custom(p[o])
#define store(p,o,x) p[o]=float_to_half_custom(x)

#define EQUILIBRIUM_BOUNDARIES
#define VOLUME_FORCE
#define MAGNETO_HYDRO
#define DEF_LOD_DEPTH 2u
#define DEF_NUM_LOD 73u
#define DEF_NUM_LOD_OWN 73u
//These defines are for code completion only and are removed from the code before compilation 
#define EndTempDefines%

// Helper functions 
inline float sq(const float x) {
	return x*x;
}
inline uint uint_sq(const uint x) {
	return x*x;
}
inline float cb(const float x) {
	return x*x*x;
}
ushort float_to_half_custom(const float x) { // custom 16-bit floating-point format, 1-4-11, exp-15, +-1.99951168, +-6.10351562E-5, +-2.98023224E-8, 3.612 digits
	const uint b = as_uint(x)+0x00000800; // round-to-nearest-even: add last bit after truncated mantissa
	const uint e = (b&0x7F800000)>>23; // exponent
	const uint m = b&0x007FFFFF; // mantissa; in line below: 0x007FF800 = 0x00800000-0x00000800 = decimal indicator flag - initial rounding
	return (b&0x80000000)>>16 | (e>112)*((((e-112)<<11)&0x7800)|m>>12) | ((e<113)&(e>100))*((((0x007FF800+m)>>(124-e))+1)>>1); // sign : normalized : denormalized (assume [-2,2])
}
float half_to_float_custom(const ushort x) { // custom 16-bit floating-point format, 1-4-11, exp-15, +-1.99951168, +-6.10351562E-5, +-2.98023224E-8, 3.612 digits
	const uint e = (x&0x7800)>>11; // exponent
	const uint m = (x&0x07FF)<<12; // mantissa
	const uint v = as_uint((float)m)>>23; // evil log2 bit hack to count leading zeros in denormalized format
	return as_float((x&0x8000)<<16 | (e!=0)*((e+112)<<23|m) | ((e==0)&(m!=0))*((v-37)<<23|((m<<(150-v))&0x007FF000))); // sign : normalized : denormalized
}
// cube of magnitude of v
float cbmagnitude(float3 v){
	return cb(sqrt(sq(v.x) + sq(v.y) + sq(v.z)));
}
inline int imax(int x, int y) {
	if (x > y) {
		return x;
	} else {
		return y;
	}
}
inline int imin(int x, int y) {
	if (x < y) {
		return x;
	} else {
		return y;
	}
}
// x to the power of DEF_DIMENSIONS
inline int to_d(int x) {
	#if defined(D2Q9)
	return x * x;
	#else
	return x * x * x;
	#endif
}
// Atomic float addition implementations for various platforms 
inline void atomic_add_f(volatile __global float* addr, const float val) {
	#if defined(cl_nv_pragma_unroll) // use hardware-supported atomic addition on Nvidia GPUs with inline PTX assembly
		float ret; asm volatile("atom.global.add.f32 %0,[%1],%2;":"=f"(ret):"l"(addr),"f"(val):"memory");
	#elif defined(__opencl_c_ext_fp32_global_atomic_add) // use hardware-supported atomic addition on some Intel GPUs
		atomic_fetch_add((volatile global atomic_float*)addr, val);
	#elif __has_builtin(__builtin_amdgcn_global_atomic_fadd_f32) // use hardware-supported atomic addition on some AMD GPUs
		__builtin_amdgcn_global_atomic_fadd_f32(addr, val);
	#else // fallback emulation: https://forums.developer.nvidia.com/t/atomicadd-float-float-atomicmul-float-float/14639/5
		float old = val; while((old=atomic_xchg(addr, atomic_xchg(addr, 0.0f)+old))!=0.0f);
	#endif
}


float3 position(const uint3 xyz) { // 3D coordinates to 3D position
	return (float3)((float)xyz.x+0.5f-0.5f*(float)DEF_NX, (float)xyz.y+0.5f-0.5f*(float)DEF_NY, (float)xyz.z+0.5f-0.5f*(float)DEF_NZ);
}
uint3 coordinates(const uint n) { // disassemble 1D index to 3D coordinates (n -> x,y,z)
	const uint t = n%(DEF_NX*DEF_NY);
	return (uint3)(t%DEF_NX, t/DEF_NX, n/(DEF_NX*DEF_NY)); // n = x+(y+z*Ny)*Nx
}
uint3 coordinates_sl(const uint n, const uint nx, const uint ny) { // disassemble 1D index and side lenghts to 3D coordinates (n -> x,y,z)
	const uint t = n%(nx*ny);
	return (uint3)(t%nx, t/nx, n/(nx*ny)); // n = x+(y+z*Ny)*Nx
}
uint index(const uint3 xyz) { // assemble 1D index from 3D coordinates (x,y,z -> n)
	return xyz.x+(xyz.y+xyz.z*DEF_NY)*DEF_NX; // n = x+(y+z*Ny)*Nx
}
bool is_halo(const uint n) {
	const uint3 xyz = coordinates(n);
	return ((DEF_DX>1u)&(xyz.x==0u||xyz.x>=DEF_NX-1u))||((DEF_DY>1u)&(xyz.y==0u||xyz.y>=DEF_NY-1u))||((DEF_DZ>1u)&(xyz.z==0u||xyz.z>=DEF_NZ-1u));
}
bool is_halo_q(const uint3 xyz) {
	return ((DEF_DX>1u)&(xyz.x==0u||xyz.x>=DEF_NX-2u))||((DEF_DY>1u)&(xyz.y==0u||xyz.y>=DEF_NY-2u))||((DEF_DZ>1u)&(xyz.z==0u||xyz.z>=DEF_NZ-2u)); // halo data is kept up-to-date, so allow using halo data for rendering
}
ulong index_f(const uint n, const uint i) { // 64-bit indexing (maximum 2^32 lattice points (1624^3 lattice resolution, 225GB)
	return (ulong)i*DEF_N+(ulong)n; // SoA (229% faster on GPU)
}
void calculate_f_eq(const float rho, float ux, float uy, float uz, float* feq) {
	const float c3=-3.0f*(sq(ux)+sq(uy)+sq(uz)), rhom1=rho-1.0f; // c3 = -2*sq(u)/(2*sq(c)), rhom1 is arithmetic optimization to minimize digit extinction
	ux *= 3.0f;
	uy *= 3.0f;
	uz *= 3.0f;
	feq[ 0] = DEF_W0*fma(rho, 0.5f*c3, rhom1); // 000 (identical for all velocity sets)
	#if defined(D2Q9)
	const float u0=ux+uy, u1=ux-uy; // these pre-calculations make manual unrolling require less FLOPs
	const float rhos=DEF_WS*rho, rhoe=DEF_WE*rho, rhom1s=DEF_WS*rhom1, rhom1e=DEF_WE*rhom1;
	feq[ 1] = fma(rhos, fma(0.5f, fma(ux, ux, c3), ux), rhom1s); feq[ 2] = fma(rhos, fma(0.5f, fma(ux, ux, c3), -ux), rhom1s); // +00 -00
	feq[ 3] = fma(rhos, fma(0.5f, fma(uy, uy, c3), uy), rhom1s); feq[ 4] = fma(rhos, fma(0.5f, fma(uy, uy, c3), -uy), rhom1s); // 0+0 0-0
	feq[ 5] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), u0), rhom1e); feq[ 6] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), -u0), rhom1e); // ++0 --0
	feq[ 7] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), u1), rhom1e); feq[ 8] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), -u1), rhom1e); // +-0 -+0
	#elif defined(D3Q15)
	const float u0=ux+uy+uz, u1=ux+uy-uz, u2=ux-uy+uz, u3=-ux+uy+uz;
	const float rhos=DEF_WS*rho, rhoc=DEF_WC*rho, rhom1s=DEF_WS*rhom1, rhom1c=DEF_WC*rhom1;
	feq[ 1] = fma(rhos, fma(0.5f, fma(ux, ux, c3), ux), rhom1s); feq[ 2] = fma(rhos, fma(0.5f, fma(ux, ux, c3), -ux), rhom1s); // +00 -00
	feq[ 3] = fma(rhos, fma(0.5f, fma(uy, uy, c3), uy), rhom1s); feq[ 4] = fma(rhos, fma(0.5f, fma(uy, uy, c3), -uy), rhom1s); // 0+0 0-0
	feq[ 5] = fma(rhos, fma(0.5f, fma(uz, uz, c3), uz), rhom1s); feq[ 6] = fma(rhos, fma(0.5f, fma(uz, uz, c3), -uz), rhom1s); // 00+ 00-
	feq[ 7] = fma(rhoc, fma(0.5f, fma(u0, u0, c3), u0), rhom1c); feq[ 8] = fma(rhoc, fma(0.5f, fma(u0, u0, c3), -u0), rhom1c); // +++ ---
	feq[ 9] = fma(rhoc, fma(0.5f, fma(u1, u1, c3), u1), rhom1c); feq[10] = fma(rhoc, fma(0.5f, fma(u1, u1, c3), -u1), rhom1c); // ++- --+
	feq[11] = fma(rhoc, fma(0.5f, fma(u2, u2, c3), u2), rhom1c); feq[12] = fma(rhoc, fma(0.5f, fma(u2, u2, c3), -u2), rhom1c); // +-+ -+-
	feq[13] = fma(rhoc, fma(0.5f, fma(u3, u3, c3), u3), rhom1c); feq[14] = fma(rhoc, fma(0.5f, fma(u3, u3, c3), -u3), rhom1c); // -++ +--
	#elif defined(D3Q19)
	const float u0=ux+uy, u1=ux+uz, u2=uy+uz, u3=ux-uy, u4=ux-uz, u5=uy-uz;
	const float rhos=DEF_WS*rho, rhoe=DEF_WE*rho, rhom1s=DEF_WS*rhom1, rhom1e=DEF_WE*rhom1;
	feq[ 1] = fma(rhos, fma(0.5f, fma(ux, ux, c3), ux), rhom1s); feq[ 2] = fma(rhos, fma(0.5f, fma(ux, ux, c3), -ux), rhom1s); // +00 -00
	feq[ 3] = fma(rhos, fma(0.5f, fma(uy, uy, c3), uy), rhom1s); feq[ 4] = fma(rhos, fma(0.5f, fma(uy, uy, c3), -uy), rhom1s); // 0+0 0-0
	feq[ 5] = fma(rhos, fma(0.5f, fma(uz, uz, c3), uz), rhom1s); feq[ 6] = fma(rhos, fma(0.5f, fma(uz, uz, c3), -uz), rhom1s); // 00+ 00-
	feq[ 7] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), u0), rhom1e); feq[ 8] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), -u0), rhom1e); // ++0 --0
	feq[ 9] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), u1), rhom1e); feq[10] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), -u1), rhom1e); // +0+ -0-
	feq[11] = fma(rhoe, fma(0.5f, fma(u2, u2, c3), u2), rhom1e); feq[12] = fma(rhoe, fma(0.5f, fma(u2, u2, c3), -u2), rhom1e); // 0++ 0--
	feq[13] = fma(rhoe, fma(0.5f, fma(u3, u3, c3), u3), rhom1e); feq[14] = fma(rhoe, fma(0.5f, fma(u3, u3, c3), -u3), rhom1e); // +-0 -+0
	feq[15] = fma(rhoe, fma(0.5f, fma(u4, u4, c3), u4), rhom1e); feq[16] = fma(rhoe, fma(0.5f, fma(u4, u4, c3), -u4), rhom1e); // +0- -0+
	feq[17] = fma(rhoe, fma(0.5f, fma(u5, u5, c3), u5), rhom1e); feq[18] = fma(rhoe, fma(0.5f, fma(u5, u5, c3), -u5), rhom1e); // 0+- 0-+
	#elif defined(D3Q27)
	const float u0=ux+uy, u1=ux+uz, u2=uy+uz, u3=ux-uy, u4=ux-uz, u5=uy-uz, u6=ux+uy+uz, u7=ux+uy-uz, u8=ux-uy+uz, u9=-ux+uy+uz;
	const float rhos=DEF_WS*rho, rhoe=DEF_WE*rho, rhoc=DEF_WC*rho, rhom1s=DEF_WS*rhom1, rhom1e=DEF_WE*rhom1, rhom1c=DEF_WC*rhom1;
	feq[ 1] = fma(rhos, fma(0.5f, fma(ux, ux, c3), ux), rhom1s); feq[ 2] = fma(rhos, fma(0.5f, fma(ux, ux, c3), -ux), rhom1s); // +00 -00
	feq[ 3] = fma(rhos, fma(0.5f, fma(uy, uy, c3), uy), rhom1s); feq[ 4] = fma(rhos, fma(0.5f, fma(uy, uy, c3), -uy), rhom1s); // 0+0 0-0
	feq[ 5] = fma(rhos, fma(0.5f, fma(uz, uz, c3), uz), rhom1s); feq[ 6] = fma(rhos, fma(0.5f, fma(uz, uz, c3), -uz), rhom1s); // 00+ 00-
	feq[ 7] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), u0), rhom1e); feq[ 8] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), -u0), rhom1e); // ++0 --0
	feq[ 9] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), u1), rhom1e); feq[10] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), -u1), rhom1e); // +0+ -0-
	feq[11] = fma(rhoe, fma(0.5f, fma(u2, u2, c3), u2), rhom1e); feq[12] = fma(rhoe, fma(0.5f, fma(u2, u2, c3), -u2), rhom1e); // 0++ 0--
	feq[13] = fma(rhoe, fma(0.5f, fma(u3, u3, c3), u3), rhom1e); feq[14] = fma(rhoe, fma(0.5f, fma(u3, u3, c3), -u3), rhom1e); // +-0 -+0
	feq[15] = fma(rhoe, fma(0.5f, fma(u4, u4, c3), u4), rhom1e); feq[16] = fma(rhoe, fma(0.5f, fma(u4, u4, c3), -u4), rhom1e); // +0- -0+
	feq[17] = fma(rhoe, fma(0.5f, fma(u5, u5, c3), u5), rhom1e); feq[18] = fma(rhoe, fma(0.5f, fma(u5, u5, c3), -u5), rhom1e); // 0+- 0-+
	feq[19] = fma(rhoc, fma(0.5f, fma(u6, u6, c3), u6), rhom1c); feq[20] = fma(rhoc, fma(0.5f, fma(u6, u6, c3), -u6), rhom1c); // +++ ---
	feq[21] = fma(rhoc, fma(0.5f, fma(u7, u7, c3), u7), rhom1c); feq[22] = fma(rhoc, fma(0.5f, fma(u7, u7, c3), -u7), rhom1c); // ++- --+
	feq[23] = fma(rhoc, fma(0.5f, fma(u8, u8, c3), u8), rhom1c); feq[24] = fma(rhoc, fma(0.5f, fma(u8, u8, c3), -u8), rhom1c); // +-+ -+-
	feq[25] = fma(rhoc, fma(0.5f, fma(u9, u9, c3), u9), rhom1c); feq[26] = fma(rhoc, fma(0.5f, fma(u9, u9, c3), -u9), rhom1c); // -++ +--
	#endif
}
void calculate_rho_u(const float* f, float* rhon, float* uxn, float* uyn, float* uzn) {
	float rho=f[0], ux, uy, uz;
	for(uint i=1u; i<DEF_VELOCITY_SET; i++) rho += f[i]; // calculate density from fi
	rho += 1.0f; // add 1.0f last to avoid digit extinction effects when summing up fi (perturbation method / DDF-shifting)
	#if defined(D2Q9)
	ux = f[1]-f[2]+f[5]-f[6]+f[7]-f[8]; // calculate velocity from fi (alternating + and - for best accuracy)
	uy = f[3]-f[4]+f[5]-f[6]+f[8]-f[7];
	uz = 0.0f;
	#elif defined(D3Q15)
	ux = f[ 1]-f[ 2]+f[ 7]-f[ 8]+f[ 9]-f[10]+f[11]-f[12]+f[14]-f[13]; // calculate velocity from fi (alternating + and - for best accuracy)
	uy = f[ 3]-f[ 4]+f[ 7]-f[ 8]+f[ 9]-f[10]+f[12]-f[11]+f[13]-f[14];
	uz = f[ 5]-f[ 6]+f[ 7]-f[ 8]+f[10]-f[ 9]+f[11]-f[12]+f[13]-f[14];
	#elif defined(D3Q19)
	ux = f[ 1]-f[ 2]+f[ 7]-f[ 8]+f[ 9]-f[10]+f[13]-f[14]+f[15]-f[16]; // calculate velocity from fi (alternating + and - for best accuracy)
	uy = f[ 3]-f[ 4]+f[ 7]-f[ 8]+f[11]-f[12]+f[14]-f[13]+f[17]-f[18];
	uz = f[ 5]-f[ 6]+f[ 9]-f[10]+f[11]-f[12]+f[16]-f[15]+f[18]-f[17];
	#elif defined(D3Q27)
	ux = f[ 1]-f[ 2]+f[ 7]-f[ 8]+f[ 9]-f[10]+f[13]-f[14]+f[15]-f[16]+f[19]-f[20]+f[21]-f[22]+f[23]-f[24]+f[26]-f[25]; // calculate velocity from fi (alternating + and - for best accuracy)
	uy = f[ 3]-f[ 4]+f[ 7]-f[ 8]+f[11]-f[12]+f[14]-f[13]+f[17]-f[18]+f[19]-f[20]+f[21]-f[22]+f[24]-f[23]+f[25]-f[26];
	uz = f[ 5]-f[ 6]+f[ 9]-f[10]+f[11]-f[12]+f[16]-f[15]+f[18]-f[17]+f[19]-f[20]+f[22]-f[21]+f[23]-f[24]+f[25]-f[26];
	#endif
	*rhon = rho;
	*uxn = ux/rho;
	*uyn = uy/rho;
	*uzn = uz/rho;
} // calculate_rho_u
void load_f(const uint n, float* fhn, const global fpxx* fi, const uint* j, const ulong t) {
	fhn[0] = load(fi, index_f(n, 0u)); // Esoteric-Pull
	for(uint i=1u; i<DEF_VELOCITY_SET; i+=2u) {
		fhn[i   ] = load(fi, index_f(n   , t%2ul ? i	: i+1u));
		fhn[i+1u] = load(fi, index_f(j[i], t%2ul ? i+1u : i   ));
	}
}
void store_f(const uint n, const float* fhn, global fpxx* fi, const uint* j, const ulong t) {
	store(fi, index_f(n, 0u), fhn[0]); // Esoteric-Pull
	for(uint i=1u; i<DEF_VELOCITY_SET; i+=2u) {
		store(fi, index_f(j[i], t%2ul ? i+1u : i   ), fhn[i   ]);
		store(fi, index_f(n   , t%2ul ? i	 : i+1u), fhn[i+1u]);
	}
}
void calculate_indices(const uint n, uint* x0, uint* xp, uint* xm, uint* y0, uint* yp, uint* ym, uint* z0, uint* zp, uint* zm) {
	const uint3 xyz = coordinates(n);
	*x0 =   xyz.x; // pre-calculate indices (periodic boundary conditions)
	*xp =  (xyz.x		+1u)%DEF_NX;
	*xm =  (xyz.x+DEF_NX-1u)%DEF_NX;
	*y0 =   xyz.y					*DEF_NX;
	*yp = ((xyz.y		+1u)%DEF_NY)*DEF_NX;
	*ym = ((xyz.y+DEF_NY-1u)%DEF_NY)*DEF_NX;
	*z0 =   xyz.z					*DEF_NY*DEF_NX;
	*zp = ((xyz.z		+1u)%DEF_NZ)*DEF_NY*DEF_NX;
	*zm = ((xyz.z+DEF_NZ-1u)%DEF_NZ)*DEF_NY*DEF_NX;
}
void neighbors(const uint n, uint* j) {
	uint x0, xp, xm, y0, yp, ym, z0, zp, zm;
	calculate_indices(n, &x0, &xp, &xm, &y0, &yp, &ym, &z0, &zp, &zm);
	j[0] = n;
	#if defined(D2Q9)
	j[ 1] = xp+y0; j[ 2] = xm+y0; // +00 -00
	j[ 3] = x0+yp; j[ 4] = x0+ym; // 0+0 0-0
	j[ 5] = xp+yp; j[ 6] = xm+ym; // ++0 --0
	j[ 7] = xp+ym; j[ 8] = xm+yp; // +-0 -+0
	#elif defined(D3Q15)
	j[ 1] = xp+y0+z0; j[ 2] = xm+y0+z0; // +00 -00
	j[ 3] = x0+yp+z0; j[ 4] = x0+ym+z0; // 0+0 0-0
	j[ 5] = x0+y0+zp; j[ 6] = x0+y0+zm; // 00+ 00-
	j[ 7] = xp+yp+zp; j[ 8] = xm+ym+zm; // +++ ---
	j[ 9] = xp+yp+zm; j[10] = xm+ym+zp; // ++- --+
	j[11] = xp+ym+zp; j[12] = xm+yp+zm; // +-+ -+-
	j[13] = xm+yp+zp; j[14] = xp+ym+zm; // -++ +--
	#elif defined(D3Q19)
	j[ 1] = xp+y0+z0; j[ 2] = xm+y0+z0; // +00 -00
	j[ 3] = x0+yp+z0; j[ 4] = x0+ym+z0; // 0+0 0-0
	j[ 5] = x0+y0+zp; j[ 6] = x0+y0+zm; // 00+ 00-
	j[ 7] = xp+yp+z0; j[ 8] = xm+ym+z0; // ++0 --0
	j[ 9] = xp+y0+zp; j[10] = xm+y0+zm; // +0+ -0-
	j[11] = x0+yp+zp; j[12] = x0+ym+zm; // 0++ 0--
	j[13] = xp+ym+z0; j[14] = xm+yp+z0; // +-0 -+0
	j[15] = xp+y0+zm; j[16] = xm+y0+zp; // +0- -0+
	j[17] = x0+yp+zm; j[18] = x0+ym+zp; // 0+- 0-+
	#elif defined(D3Q27)
	j[ 1] = xp+y0+z0; j[ 2] = xm+y0+z0; // +00 -00
	j[ 3] = x0+yp+z0; j[ 4] = x0+ym+z0; // 0+0 0-0
	j[ 5] = x0+y0+zp; j[ 6] = x0+y0+zm; // 00+ 00-
	j[ 7] = xp+yp+z0; j[ 8] = xm+ym+z0; // ++0 --0
	j[ 9] = xp+y0+zp; j[10] = xm+y0+zm; // +0+ -0-
	j[11] = x0+yp+zp; j[12] = x0+ym+zm; // 0++ 0--
	j[13] = xp+ym+z0; j[14] = xm+yp+z0; // +-0 -+0
	j[15] = xp+y0+zm; j[16] = xm+y0+zp; // +0- -0+
	j[17] = x0+yp+zm; j[18] = x0+ym+zp; // 0+- 0-+
	j[19] = xp+yp+zp; j[20] = xm+ym+zm; // +++ ---
	j[21] = xp+yp+zm; j[22] = xm+ym+zp; // ++- --+
	j[23] = xp+ym+zp; j[24] = xm+yp+zm; // +-+ -+-
	j[25] = xm+yp+zp; j[26] = xp+ym+zm; // -++ +--
	#endif
} //neighbors
float3 load_u(const uint n, const global float* u) {
	return (float3)(u[n], u[DEF_N+(ulong)n], u[2ul*DEF_N+(ulong)n]);
}
float calculate_Q_cached(const float3 u0, const float3 u1, const float3 u2, const float3 u3, const float3 u4, const float3 u5) { // Q-criterion
	const float duxdx=u0.x-u1.x, duydx=u0.y-u1.y, duzdx=u0.z-u1.z; // du/dx = (u2-u0)/2
	const float duxdy=u2.x-u3.x, duydy=u2.y-u3.y, duzdy=u2.z-u3.z;
	const float duxdz=u4.x-u5.x, duydz=u4.y-u5.y, duzdz=u4.z-u5.z;
	const float omega_xy=duxdy-duydx, omega_xz=duxdz-duzdx, omega_yz=duydz-duzdy; // antisymmetric tensor, omega_xx = omega_yy = omega_zz = 0
	const float s_xx2=duxdx, s_yy2=duydy, s_zz2=duzdz; // s_xx2 = s_xx/2, s_yy2 = s_yy/2, s_zz2 = s_zz/2
	const float s_xy=duxdy+duydx, s_xz=duxdz+duzdx, s_yz=duydz+duzdy; // symmetric tensor
	const float omega2 = sq(omega_xy)+sq(omega_xz)+sq(omega_yz); // ||omega||_2^2
	const float s2 = 2.0f*(sq(s_xx2)+sq(s_yy2)+sq(s_zz2))+sq(s_xy)+sq(s_xz)+sq(s_yz); // ||s||_2^2
	return 0.25f*(omega2-s2); // Q = 1/2*(||omega||_2^2-||s||_2^2), addidional factor 1/2 from cental finite differences of velocity
} // calculate_Q_cached()
float calculate_Q(const uint n, const global float* u) { // Q-criterion
	uint x0, xp, xm, y0, yp, ym, z0, zp, zm;
	calculate_indices(n, &x0, &xp, &xm, &y0, &yp, &ym, &z0, &zp, &zm);
	uint j[6];
	j[0] = xp+y0+z0; j[1] = xm+y0+z0; // +00 -00
	j[2] = x0+yp+z0; j[3] = x0+ym+z0; // 0+0 0-0
	j[4] = x0+y0+zp; j[5] = x0+y0+zm; // 00+ 00-
	return calculate_Q_cached(load_u(j[0], u), load_u(j[1], u), load_u(j[2], u), load_u(j[3], u), load_u(j[4], u), load_u(j[5], u));
} // calculate_Q()
float c(const uint i) { // avoid constant keyword by encapsulating data in function which gets inlined by compiler
	const float c[3u*DEF_VELOCITY_SET] = {
	#if defined(D2Q9)
		0, 1,-1, 0, 0, 1,-1, 1,-1, // x
		0, 0, 0, 1,-1, 1,-1,-1, 1, // y
		0, 0, 0, 0, 0, 0, 0, 0, 0  // z
	#elif defined(D3Q15)
		0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, // x
		0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1,-1, 1, 1,-1, // y
		0, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1  // z
	#elif defined(D3Q19)
		0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, // x
		0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, // y
		0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1  // z
	#elif defined(D3Q27)
		0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, // x
		0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, // y
		0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1  // z
	#endif // D3Q27
	};
	return c[i];
}
float w(const uint i) { // avoid constant keyword by encapsulating data in function which gets inlined by compiler
	const float w[DEF_VELOCITY_SET] = { DEF_W0, // velocity set weights
	#if defined(D2Q9)
		DEF_WS, DEF_WS, DEF_WS, DEF_WS, DEF_WE, DEF_WE, DEF_WE, DEF_WE
	#elif defined(D3Q15)
		DEF_WS, DEF_WS, DEF_WS, DEF_WS, DEF_WS, DEF_WS,
		DEF_WC, DEF_WC, DEF_WC, DEF_WC, DEF_WC, DEF_WC, DEF_WC, DEF_WC
	#elif defined(D3Q19)
		DEF_WS, DEF_WS, DEF_WS, DEF_WS, DEF_WS, DEF_WS,
		DEF_WE, DEF_WE, DEF_WE, DEF_WE, DEF_WE, DEF_WE, DEF_WE, DEF_WE, DEF_WE, DEF_WE, DEF_WE, DEF_WE
	#elif defined(D3Q27)
		DEF_WS, DEF_WS, DEF_WS, DEF_WS, DEF_WS, DEF_WS,
		DEF_WE, DEF_WE, DEF_WE, DEF_WE, DEF_WE, DEF_WE, DEF_WE, DEF_WE, DEF_WE, DEF_WE, DEF_WE, DEF_WE,
		DEF_WC, DEF_WC, DEF_WC, DEF_WC, DEF_WC, DEF_WC, DEF_WC, DEF_WC
	#endif
	};
	return w[i];
}
#ifdef VOLUME_FORCE
void calculate_forcing_terms(const float ux, const float uy, const float uz, const float fx, const float fy, const float fz, float* Fin) { // calculate volume force terms Fin from velocity field (Guo forcing, Krueger p.233f)
	#ifdef D2Q9
		const float uF = -0.33333334f*fma(ux, fx, uy*fy); // 2D
	#else
		const float uF = -0.33333334f*fma(ux, fx, fma(uy, fy, uz*fz)); // 3D
	#endif
	Fin[0] = 9.0f*DEF_W0*uF ; // 000 (identical for all velocity sets)
	for(uint i=1u; i<DEF_VELOCITY_SET; i++) { // loop is entirely unrolled by compiler, no unnecessary FLOPs are happening
		Fin[i] = 9.0f*w(i)*fma(c(i)*fx+c(DEF_VELOCITY_SET+i)*fy+c(2u*DEF_VELOCITY_SET+i)*fz, c(i)*ux+c(DEF_VELOCITY_SET+i)*uy+c(2u*DEF_VELOCITY_SET+i)*uz+0.33333334f, uF);
	}
}
#endif // VOLUME_FORCE

#ifdef MAGNETO_HYDRO
// Charge advection
void neighbors_a(const uint n, uint* j7) { // calculate advection property neighbor indices
	uint x0, xp, xm, y0, yp, ym, z0, zp, zm;
	calculate_indices(n, &x0, &xp, &xm, &y0, &yp, &ym, &z0, &zp, &zm);
	j7[0] = n;
	j7[1] = xp+y0+z0; j7[2] = xm+y0+z0; // +00 -00
	j7[3] = x0+yp+z0; j7[4] = x0+ym+z0; // 0+0 0-0
	j7[5] = x0+y0+zp; j7[6] = x0+y0+zm; // 00+ 00-
}
void calculate_a_eq(const float Q, const float ux, const float uy, const float uz, float* qeq) { // calculate a_equilibrium from density and velocity field (perturbation method / DDF-shifting)
	const float wsT4=0.5f*Q, wsTm1=0.125f*(Q-1.0f); // 0.125f*Q*4.0f (straight directions in D3Q7), wsTm1 is arithmetic optimization to minimize digit extinction, lattice speed of sound is 1/2 for D3Q7 and not 1/sqrt(3)
	qeq[0] = fma(0.25f, Q, -0.25f); // 000
	qeq[1] = fma(wsT4, ux, wsTm1); qeq[2] = fma(wsT4, -ux, wsTm1); // +00 -00, source: http://dx.doi.org/10.1016/j.ijheatmasstransfer.2009.11.014
	qeq[3] = fma(wsT4, uy, wsTm1); qeq[4] = fma(wsT4, -uy, wsTm1); // 0+0 0-0
	qeq[5] = fma(wsT4, uz, wsTm1); qeq[6] = fma(wsT4, -uz, wsTm1); // 00+ 00-
}
void load_a(const uint n, float* qhn, const global fpxx* fqi, const uint* j7, const ulong t) {
	qhn[0] = load(fqi, index_f(n, 0u)); // Esoteric-Pull
	for(uint i=1u; i<7u; i+=2u) {
		qhn[i   ] = load(fqi, index_f(n	   , t%2ul ? i	: i+1u));
		qhn[i+1u] = load(fqi, index_f(j7[i], t%2ul ? i+1u : i   ));
	}
}
void store_a(const uint n, const float* qhn, global fpxx* fqi, const uint* j7, const ulong t) {
	store(fqi, index_f(n, 0u), qhn[0]); // Esoteric-Pull
	for(uint i=1u; i<7u; i+=2u) {
		store(fqi, index_f(j7[i], t%2ul ? i+1u : i   ), qhn[i   ]);
		store(fqi, index_f(n	, t%2ul ? i	: i+1u), qhn[i+1u]);
	}
}

// Ionization crosssection
float calculate_sigma_i(const float v) {
	// eV = 9.10938356e-31/(2*1.6021766208e-19) * v^2 = 0.00000000000284281503 * v^2 = DEF_KVEV * v^2
	float ev = DEF_KVEV * sq(v);
	#if defined(PROP_XE) // xenon propellant
		return 6.5f*pow((ev-12.f)/ev, 1.2f) // ionization crosssection for xenon scaled by 10^20, https://descanso.jpl.nasa.gov/SciTechBook/series1/Goebel_AppD_IonizationCross.pdf
	#else
		return 0.0f;
	#endif
}

// LOD handling
// Get 1D LOD index from cell index n and depth d (No offset for previous lod depths)
uint lod_index(const uint n, const uint d) {
	const uint3 c = coordinates(n);
	const uint nd = (1<<d); // Number of lods on each axis
	// TODO: Arithmetic optimization
	const uint x = c.x / (DEF_NX / nd);
	const uint y = c.y / (DEF_NY / nd);
	const uint z = c.z / (DEF_NZ / nd);
	return x + (y + z * nd) * nd;
}
// Size of LODs at the specified depth d
float lod_s(const uint d) {
	const uint nd = (1<<d); // Number of lods on each axis
	return (float)((DEF_NX / nd) * (DEF_NY / nd) * (DEF_NZ / nd));
}
// float coords of the center of an LOD from 1D LOD index n and specified depth d
float3 lod_coordinates(const uint n, const uint d) { // 
	const uint nd = (1<<d); // Number of lods on each axis
	const float dsx = (float)(DEF_NX / nd);
	const float dsy = (float)(DEF_NY / nd);
	const float dsz = (float)(DEF_NZ / nd);
	const uint t = n%(nd*nd);
	return (float3)((float)(t%nd)*dsx+(0.5f*dsx), (float)(t/nd)*dsy+(0.5f*dsy), (float)(n/(nd*nd))*dsz+(0.5f*dsz)); // n = x+(y+z*Ny)*Nx
}
#endif
#ifdef SUBGRID_ECR
float mag_v(const uint n, const global float* V) { // Magnitude of a vector field at n
	return sqrt(sq(V[n]) + sq(V[DEF_N+(ulong)n]) + sq(V[2ul*DEF_N+(ulong)n]));
}
float3 grad_mag_v(const uint n, const global float* V) { // Gradient of the magnitude of a vector field at n
	uint3 c = coordinates(n);
	return (float3)(
		mag_v(index(c + (uint3)(1, 0, 0)), V) - mag_v(index(c - (uint3)(1, 0, 0)), V) / 2.0f,
		mag_v(index(c + (uint3)(0, 1, 0)), V) - mag_v(index(c - (uint3)(0, 1, 0)), V) / 2.0f,
		mag_v(index(c + (uint3)(0, 0, 1)), V) - mag_v(index(c - (uint3)(0, 0, 1)), V) / 2.0f,
	);
}
#endif // SUBGRID_ECR

// Simulation kernel functions
__kernel void stream_collide(global fpxx* fi, global float* rho, global float* u, global uchar* flags, const ulong t, const float fx, const float fy, const float fz 
#ifdef FORCE_FIELD
, const global float* F 
#endif // FORCE_FIELD
#ifdef MAGNETO_HYDRO
, const global float* E_dyn	// dynamic electric field
, const global float* B_dyn	// dynamic magnetic flux density
, global fpxx* fqi		// charge property of gas as ddfs
, global fpxx* ei		// electron gas ddfs
, global float* Q		// cell charge
, global float* QU_lod	// Level-of-detail for charge und velocity 
#endif // MAGNETO_HYDRO
#ifdef SUBGRID_ECR
, const global float* E_var // Oscillating electric field, actually static, contained information is direction of oscillation + magnitude of field strenght
, global fpxx* eti          // electron temperature ddfs
, global float* Et          // electron temperature field
, const float ecrf          // ECR frequency
#endif // SUBGRID_ECR
) {
	const uint n = get_global_id(0); // n = x+(y+z*Ny)*Nx
	if(n>=(uint)DEF_N||is_halo(n)) return; // don't execute stream_collide() on halo
	const uchar flagsn = flags[n]; // cache flags[n] for multiple readings
	const uchar flagsn_bo=flagsn&TYPE_BO; // extract boundary and surface flags
	if(flagsn_bo==TYPE_S) return; // if cell is solid boundary or gas, just return

	uint j[DEF_VELOCITY_SET]; // neighbor indices
	neighbors(n, j); // calculate neighbor indices

	float fhn[DEF_VELOCITY_SET]; // local DDFs
	load_f(n, fhn, fi, j, t); // perform streaming (part 2)

	ulong nxi=(ulong)n, nyi=DEF_N+(ulong)n, nzi=2ul*DEF_N+(ulong)n; // n indecies for x, y and z components

	float rhon, uxn, uyn, uzn; // calculate local density and velocity for collision

	#ifndef EQUILIBRIUM_BOUNDARIES // EQUILIBRIUM_BOUNDARIES
		calculate_rho_u(fhn, &rhon, &uxn, &uyn, &uzn); // calculate density and velocity fields from fi
	#else
		if(flagsn_bo==TYPE_E) {
			rhon = rho[n]; // apply preset velocity/density
			uxn  = u[nxi];
			uyn  = u[nyi];
			uzn  = u[nzi];
		} else {
			calculate_rho_u(fhn, &rhon, &uxn, &uyn, &uzn); // calculate density and velocity fields from fi
		}
	#endif // EQUILIBRIUM_BOUNDARIES

	float fxn=fx, fyn=fy, fzn=fz; // force starts as constant volume force, can be modified before call of calculate_forcing_terms(...)

	float Fin[DEF_VELOCITY_SET]; // forcing terms, are used for ei too if MHD is enabled 
	float feq[DEF_VELOCITY_SET]; // equilibrium DDFs, are used for ei too if MHD is enabled 
	float w = DEF_W; // LBM relaxation rate w = dt/tau = dt/(nu/c^2+dt/2) = 1/(3*nu+1/2)
	#ifdef VOLUME_FORCE
	const float c_tau = fma(w, -0.5f, 1.0f);
	#endif // VOLUME_FORCE

	#ifdef FORCE_FIELD
	{ // separate block to avoid variable name conflicts
		fxn += F[nxi]; // apply force field
		fyn += F[nyi];
		fzn += F[nzi];
	}
	#endif

	#ifdef MAGNETO_HYDRO
		/* -------- Cache fields -------- */
		float3 Bn = {B_dyn[nxi], B_dyn[nyi], B_dyn[nzi]}; // Cache dynamic fields for multiple readings
		float3 En = {E_dyn[nxi], E_dyn[nyi], E_dyn[nzi]};		


		/* ---- Electron gas part 1 ----- */
		float ehn[DEF_VELOCITY_SET]; // local DDFs
		load_f(n, ehn, ei, j, t); // perform streaming (part 2)
		float rhon_e, uxn_e, uyn_e, uzn_e; // calculate local density and velocity for collision
		calculate_rho_u(ehn, &rhon_e, &uxn_e, &uyn_e, &uzn_e); // calculate (charge) density and velocity fields from ei


		/* --- Gas charge advection 1 --- */
		// Advection of charge. Cell charge is stored in charge ddfs 'fqi' for advection with 'fi'.
		uint j7[7]; // neighbors of D3Q7 subset
		neighbors_a(n, j7);
		float qhn[7]; // read from qA and stream to gh (D3Q7 subset, periodic boundary conditions)
		load_a(n, qhn, fqi, j7, t); // perform streaming (part 2)
		float rhon_q = 0.0f;
		for(uint i=0u; i<7u; i++) rhon_q += qhn[i]; // calculate charge from q
		rhon_q += 1.0f; // add 1.0f last to avoid digit extinction effects when summing up fqi (perturbation method / DDF-shifting)


		/* --------- Subgrid ECR -------- */
		#ifdef SUBGRID_ECR
			// The SUBGRID ECR extension provides electron temperature, ionization modelling and electron 
			// cyclotron resonance heating. The electron temperature is assumed to be proportional to
			// the gyrating motion of the electrons in a magnetic field, the speed in the electron gas ddfs
			// represents a guiding center drift.


			/* --- Electron temperature 1 --- */
			float ethn[7]; // read from gA and stream to gh (D3Q7 subset, periodic boundary conditions)
			load_a(n, ethn, eti, j7, t); // perform streaming (part 2)
			float Etn;
			//if(flagsn&TYPE_T) {
			//	Etn = T[n]; // apply preset temperature
			//} else {
			Etn = 0.0f;
			for(uint i=0u; i<7u; i++) Etn += ethn[i]; // calculate temperature from g
			Etn += 1.0f; // add 1.0f last to avoid digit extinction effects when summing up gi (perturbation method / DDF-shifting)
			//}


			/* --------- ECR heating -------- */
			float3 Env = {E_var[nxi], E_var[nyi], E_var[nzi]}; // Oscillating electric field as vector
			// calculate energy absorbtion under the possibility that frequency does not fulfill ECR condition
			float f_c = length(Bn) * (1.0f / (DEF_KME * 2.0f * M_PI_F)); // The cyclotron frequency needed to fulfill ECR condition at current cell
			float rel_absorbtion = 1.0f / (1.0f + sq(ecrf / (0.03125 * f_c) - 32)); // dampening value 0.03125/32, computed through simulation and best fit aproximation
			float Env_mag = length(Env - (dot(Env, Bn) / sq(length(B)))*Bn); // Magnitude of oscillating electric field components perpendicular to B, only these have effect for ECR
			Etn += DEF_KEABS / DEF_KKBME * rhon_e / DEF_KKGE  * sq(Env_mag); // Energy increase W = (e² / 2m_e²) * m_e_all * E²

			// Drift of gyrating electrons in magnetic field. We assume all kinetic energy from electron
			// temperature is directed perpendicular to the magnetic field. https://doi.org/10.1007/978-3-662-55236-0_1 2.2.4
			// dv_∥/dt = (-0.5 * v_⟂^2 ∇ B)/B = (-1.5 * k_B * T_e * ∇ B)/(m_e * B)
			float3 delta_u_par = (DEF_KKBME * Etn * grad_mag_v(n, B_dyn)) / length(Bn); // time step length = 1
			uxn_e += delta_u_par.x;
			uyn_e += delta_u_par.y;
			uzn_e += delta_u_par.z;
			Etn -= length(delta_u_par) / DEF_KKBME // Temperature reduces proportional to the drift velocity


			/* --- Electron temperature 2 --- */
			float eteq[7]; // cache f_equilibrium[n]
			calculate_a_eq(Etn, uxn_e, uyn_e, uzn_e, eteq); // calculate equilibrium DDFs
			//if(flagsn&TYPE_T) {
			//	for(uint i=0u; i<7u; i++) ethn[i] = eteq[i]; // just write eteq to ethn (no collision)
			//} else {
			#ifdef UPDATE_FIELDS
				Et[n] = Etn; // update temperature field
			#endif // UPDATE_FIELDS
			for(uint i=0u; i<7u; i++) ethn[i] = fma(1.0f-def_w_T, ethn[i], def_w_T*eteq[i]); // perform collision
			//}
			store_a(n, ethn, gi, j7, t); // perform streaming (part 1)


			/* --------- Ionization --------- */
			//float e_rhon_m = rhon_e * DEF_KKGE; // convert charge density to mass density,
			// TODO: Ionization
			//float v_r = sqrt(sq(uxn-uxn_e) + sq(uyn-uyn_e) + sq(uzn-uzn_e));
			// $\Delta rhon_e=(rhon/m_g)*rhon_e*v_r*sigma_i(v_r)$
			// DEF_KMG & sigma_i are scaled by a factor of 10^20 to minimize floating point imprecision
			//float delta_q_rho = ((rhon*DEF_KIMG) * (rhon_e * DEF_KKGE) * v_r * calculate_sigma_i(v_r)) / DEF_KKGE;
			float delta_q_rho = 0.0f;
			rhon_e += delta_q_rho; // Freeing of electrons through ionization adds charge/mass to the electron gas 
			rhon_q += delta_q_rho; // Ionization of neutral gas adds charge to the neutral gas
		#endif // SUBGRID_ECR


		/* --- Gas charge advection 2 --- */
		Q[n] = rhon_q-rhon_e; // update charge field
		float qeq[7]; // cache f_equilibrium[n]
		calculate_a_eq(rhon_q, uxn, uyn, uzn, qeq); // calculate equilibrium DDFs
		for(uint i=0u; i<7u; i++) qhn[i] = fma(1.0f-DEF_WQ, qhn[i], DEF_WQ*qeq[i]); // perform collision
		store_a(n, qhn, fqi, j7, t); // perform streaming (part 1)


		/* ---- Electron gas part 2 ----- */
		float3 e_fn = -rhon_e * (En + cross((float3)(uxn_e, uyn_e, uzn_e), Bn)); // F = charge * (E + (U cross B)), charge is content f ddfs
		const float rho2_e = 0.5f/(rhon_e * DEF_KKGE); // apply external volume force (Guo forcing, Krueger p.233f)
		uxn_e = clamp(fma(e_fn.x, rho2_e, uxn_e), -DEF_C, DEF_C); // limit velocity (for stability purposes)
		uyn_e = clamp(fma(e_fn.y, rho2_e, uyn_e), -DEF_C, DEF_C); // force term: F*dt/(2*rho)
		uzn_e = clamp(fma(e_fn.z, rho2_e, uzn_e), -DEF_C, DEF_C);
		calculate_forcing_terms(uxn_e, uyn_e, uzn_e, e_fn.x, e_fn.y, e_fn.z, Fin); // calculate volume force terms Fin from velocity field (Guo forcing, Krueger p.233f)
		calculate_f_eq(rhon_e, uxn_e, uyn_e, uzn_e, feq); // calculate equilibrium DDFs
		// Perform collision with SRT TODO: use correct kinematic viscosity
		for(uint i=0u; i<DEF_VELOCITY_SET; i++) Fin[i] *= c_tau;
		#ifndef EQUILIBRIUM_BOUNDARIES
			for(uint i=0u; i<DEF_VELOCITY_SET; i++) ehn[i] = fma(1.0f-w, ehn[i], fma(w, feq[i], Fin[i])); // perform collision (SRT)
		#else
			for(uint i=0u; i<DEF_VELOCITY_SET; i++) ehn[i] = flagsn_bo==TYPE_E ? feq[i] : fma(1.0f-w, ehn[i], fma(w, feq[i], Fin[i])); // perform collision (SRT)
		#endif // EQUILIBRIUM_BOUNDARIES
		store_f(n, ehn, ei, j, t); // perform streaming (part 1)


		/* ------ EM force on gas ------- */
		fxn += rhon_q * (En.x + uyn*Bn.z - uzn*Bn.y); // F = charge * (E + (U cross B))
		fyn += rhon_q * (En.y + uzn*Bn.x - uxn*Bn.z); // charge is the content of the ddf
		fzn += rhon_q * (En.z + uxn*Bn.y - uyn*Bn.x);


		/* ------ LOD construction ------ */
		#if DEF_LOD_DEPTH > 0 
			uint off = 0; // Update LOD buffer
			#if (DEF_DX>1 || DEF_DY>1 || DEF_DZ>1)  
				for (uint d = 0; d<DEF_LOD_DEPTH; d++) { off += (1<<d*DEF_DIMENSIONS); } // Multiple Domains, set offset to lower details
			#endif // Single Domain
			const uint ind = (lod_index(n, DEF_LOD_DEPTH) + off) * 4;
			const float ils = 1.0f/lod_s(DEF_LOD_DEPTH); // factor for averaging accross LODs
			atomic_add_f(&QU_lod[ind+0], rhon_q-rhon_e);
			atomic_add_f(&QU_lod[ind+1], uxn * ils);
			atomic_add_f(&QU_lod[ind+2], uyn * ils);
			atomic_add_f(&QU_lod[ind+3], uzn * ils);
		#endif
	#endif// MAGNETO_HYDRO

	#ifdef VOLUME_FORCE
		const float rho2 = 0.5f/rhon; // apply external volume force (Guo forcing, Krueger p.233f)
		uxn = clamp(fma(fxn, rho2, uxn), -DEF_C, DEF_C); // limit velocity (for stability purposes)
		uyn = clamp(fma(fyn, rho2, uyn), -DEF_C, DEF_C); // force term: F*dt/(2*rho)
		uzn = clamp(fma(fzn, rho2, uzn), -DEF_C, DEF_C);
		calculate_forcing_terms(uxn, uyn, uzn, fxn, fyn, fzn, Fin); // calculate volume force terms Fin from velocity field (Guo forcing, Krueger p.233f)
	#else // VOLUME_FORCE
		uxn = clamp(uxn, -DEF_C, DEF_C); // limit velocity (for stability purposes)
		uyn = clamp(uyn, -DEF_C, DEF_C); // force term: F*dt/(2*rho)
		uzn = clamp(uzn, -DEF_C, DEF_C);
		for(uint i=0u; i<DEF_VELOCITY_SET; i++) Fin[i] = 0.0f;
	#endif // VOLUME_FORCE


	#ifndef EQUILIBRIUM_BOUNDARIES
	#ifdef UPDATE_FIELDS
		rho[n] = rhon; // update density field
		u[nxi] = uxn; // update velocity field
		u[nyi] = uyn;
		u[nzi] = uzn;
	#endif // UPDATE_FIELDS
	#else // EQUILIBRIUM_BOUNDARIES
	#ifdef UPDATE_FIELDS
		if(flagsn_bo!=TYPE_E) { // only update fields for non-TYPE_E cells
			rho[n] = rhon; // update density field
			u[nxi] = uxn; // update velocity field
			u[nyi] = uyn;
			u[nzi] = uzn;
		}
	#endif // UPDATE_FIELDS
	#endif // EQUILIBRIUM_BOUNDARIES

	calculate_f_eq(rhon, uxn, uyn, uzn, feq); // calculate equilibrium DDFs

	#if defined(SRT) // SRT
		#ifdef VOLUME_FORCE
			for(uint i=0u; i<DEF_VELOCITY_SET; i++) Fin[i] *= c_tau;
		#endif // VOLUME_FORCE

		#ifndef EQUILIBRIUM_BOUNDARIES
			for(uint i=0u; i<DEF_VELOCITY_SET; i++) fhn[i] = fma(1.0f-w, fhn[i], fma(w, feq[i], Fin[i])); // perform collision (SRT)
		#else
			for(uint i=0u; i<DEF_VELOCITY_SET; i++) fhn[i] = flagsn_bo==TYPE_E ? feq[i] : fma(1.0f-w, fhn[i], fma(w, feq[i], Fin[i])); // perform collision (SRT)
		#endif // EQUILIBRIUM_BOUNDARIES

	#elif defined(TRT) // TRT
		const float wp = w; // TRT: inverse of "+" relaxation time
		const float wm = 1.0f/(0.1875f/(1.0f/w-0.5f)+0.5f); // TRT: inverse of "-" relaxation time wm = 1.0f/(0.1875f/(3.0f*nu)+0.5f), nu = (1.0f/w-0.5f)/3.0f;

		#ifdef VOLUME_FORCE
			const float c_taup=fma(wp, -0.25f, 0.5f), c_taum=fma(wm, -0.25f, 0.5f); // source: https://arxiv.org/pdf/1901.08766.pdf
			float Fib[DEF_VELOCITY_SET]; // F_bar
			Fib[0] = Fin[0];
			for(uint i=1u; i<DEF_VELOCITY_SET; i+=2u) {
				Fib[i   ] = Fin[i+1u];
				Fib[i+1u] = Fin[i   ];
			}
			for(uint i=0u; i<DEF_VELOCITY_SET; i++) Fin[i] = fma(c_taup, Fin[i]+Fib[i], c_taum*(Fin[i]-Fib[i]));
		#endif // VOLUME_FORCE

		float fhb[DEF_VELOCITY_SET]; // fhn in inverse directions
		float feb[DEF_VELOCITY_SET]; // feq in inverse directions
		fhb[0] = fhn[0];
		feb[0] = feq[0];
		for(uint i=1u; i<DEF_VELOCITY_SET; i+=2u) {
			fhb[i   ] = fhn[i+1u];
			fhb[i+1u] = fhn[i   ];
			feb[i   ] = feq[i+1u];
			feb[i+1u] = feq[i   ];
		}
		#ifndef EQUILIBRIUM_BOUNDARIES
			for(uint i=0u; i<DEF_VELOCITY_SET; i++) fhn[i] = fma(0.5f*wp, feq[i]-fhn[i]+feb[i]-fhb[i], fma(0.5f*wm, feq[i]-feb[i]-fhn[i]+fhb[i], fhn[i]+Fin[i])); // perform collision (TRT)
		#else // EQUILIBRIUM_BOUNDARIES
			for(uint i=0u; i<DEF_VELOCITY_SET; i++) fhn[i] = flagsn_bo==TYPE_E ? feq[i] : fma(0.5f*wp, feq[i]-fhn[i]+feb[i]-fhb[i], fma(0.5f*wm, feq[i]-feb[i]-fhn[i]+fhb[i], fhn[i]+Fin[i])); // perform collision (TRT)
		#endif // EQUILIBRIUM_BOUNDARIES
	#endif // TRT

	store_f(n, fhn, fi, j, t); // perform streaming (part 1)
} // stream_collide()

__kernel void initialize(global fpxx* fi, global float* rho, global float* u, global uchar* flags
#ifdef MAGNETO_HYDRO
, const global float* E
, const global float* B
, global float* E_dyn
, global float* B_dyn
, global fpxx* fqi
, global fpxx* ei		// electron gas ddfs
, global float* Q
#endif // MAGNETO_HYDRO
#ifdef SUBGRID_ECR
, global fpxx* eti
, const global float* Et
#endif // SUBGRID_ECR
) {
	const uint n = get_global_id(0); // n = x+(y+z*Ny)*Nx
	if(n>=(uint)DEF_N||is_halo(n)) return; // don't execute initialize() on halo
	ulong nxi=(ulong)n, nyi=DEF_N+(ulong)n, nzi=2ul*DEF_N+(ulong)n; // n indecies for x, y and z components
	uchar flagsn = flags[n];
	const uchar flagsn_bo = flagsn&TYPE_BO; // extract boundary flags
	uint j[DEF_VELOCITY_SET]; // neighbor indices
	neighbors(n, j); // calculate neighbor indices
	uchar flagsj[DEF_VELOCITY_SET]; // cache neighbor flags for multiple readings
	for(uint i=1u; i<DEF_VELOCITY_SET; i++) flagsj[i] = flags[j[i]];
	if(flagsn_bo==TYPE_S) { // cell is solid
		bool TYPE_ONLY_S = true; // has only solid neighbors
		for(uint i=1u; i<DEF_VELOCITY_SET; i++) TYPE_ONLY_S = TYPE_ONLY_S&&(flagsj[i]&TYPE_BO)==TYPE_S;
		if(TYPE_ONLY_S) {
			u[nxi] = 0.0f; // reset velocity for solid lattice points with only boundary neighbors
			u[nyi] = 0.0f;
			u[nzi] = 0.0f;
			#ifdef MAGNETO_HYDRO
			Q[n] = 0.0f;
			#endif
		}
		if(flagsn_bo==TYPE_S) {
			u[nxi] = 0.0f; // reset velocity for all solid lattice points
			u[nyi] = 0.0f;
			u[nzi] = 0.0f;
			#ifdef MAGNETO_HYDRO
			Q[n] = 0.0f;
			#endif
		}
	}
	float fe_eq[DEF_VELOCITY_SET]; // f_equilibrium, reused for e_equilibrium
	calculate_f_eq(rho[n], u[n], u[DEF_N+(ulong)n], u[2ul*DEF_N+(ulong)n], fe_eq);
	store_f(n, fe_eq, fi, j, 1ul); // write to fi

	#ifdef MAGNETO_HYDRO
		// Initialize charge ddfs
		float qeq[7]; // q_equilibrium
		calculate_a_eq(Q[n], u[n], u[DEF_N+(ulong)n], u[2ul*DEF_N+(ulong)n], qeq);
		uint j7[7]; // neighbors of D3Q7 subset
		neighbors_a(n, j7);
		store_a(n, qeq, fqi, j7, 1ul); // write to fqi. perform streaming (part 1)
		// Clear dyn with static field for recomputation
		B_dyn[nxi] = B[nxi];
		B_dyn[nyi] = B[nyi];
		B_dyn[nzi] = B[nzi];
		E_dyn[nxi] = E[nxi];
		E_dyn[nyi] = E[nyi];
		E_dyn[nzi] = E[nzi];
		// Initialize electron gas ddfs
		calculate_f_eq(Q[n], u[n], u[DEF_N+(ulong)n], u[2ul*DEF_N+(ulong)n], fe_eq);
		store_f(n, fe_eq, ei, j, 1ul); // write to fi
	#endif // MAGNETO_HYDRO
	#ifdef SUBGRID_ECR
		 // Initialize electron temperature ddfs
		calculate_a_eq(Et[n], u[n], u[DEF_N+(ulong)n], u[2ul*DEF_N+(ulong)n], qeq);
		store_a(n, qeq, eti, j7, 1ul); // write to eti. perform streaming (part 1)
	#endif // SUBGRID_ECR
} // initialize()

__kernel void update_fields(const global fpxx* fi, global float* rho, global float* u, const global uchar* flags, const ulong t, const float fx, const float fy, const float fz) {
	const uint n = get_global_id(0); // n = x+(y+z*Ny)*Nx
	if(n>=(uint)DEF_N||is_halo(n)) return; // don't execute update_fields() on halo
	const uchar flagsn = flags[n];
	const uchar flagsn_bo=flagsn&TYPE_BO; // extract boundary and surface flags
	if(flagsn_bo==TYPE_S) return; // don't update fields for boundary or gas lattice points

	uint j[DEF_VELOCITY_SET]; // neighbor indices
	neighbors(n, j); // calculate neighbor indices
	float fhn[DEF_VELOCITY_SET]; // local DDFs
	load_f(n, fhn, fi, j, t); // perform streaming (part 2)

	float rhon, uxn, uyn, uzn; // calculate local density and velocity for collision
	calculate_rho_u(fhn, &rhon, &uxn, &uyn, &uzn); // calculate density and velocity fields from fi
	float fxn=fx, fyn=fy, fzn=fz; // force starts as constant volume force, can be modified before call of calculate_forcing_terms(...)
	{
		uxn = clamp(uxn, -DEF_C, DEF_C); // limit velocity (for stability purposes)
		uyn = clamp(uyn, -DEF_C, DEF_C); // force term: F*dt/(2*rho)
		uzn = clamp(uzn, -DEF_C, DEF_C);
	}

	rho[			   n] = rhon; // update density field
	u[				   n] = uxn; // update velocity field
	u[	  DEF_N+(ulong)n] = uyn;
	u[2ul*DEF_N+(ulong)n] = uzn;
} // update_fields()

#ifdef MAGNETO_HYDRO
// Assemble lower-level lods from highest-level lods generated in stream_collide (part 1)
// Call in range LOD_DEPTH-1 ..= 0
__kernel void lod_part_2_gather(global float* lods, const uint depth) {
    uint n = get_global_id(0);

	const uint nd = (1<<depth); // n of lods on each axis for depth 
	const uint t = n%(nd*nd);
	const uint3 nlodcb = (uint3)(t%nd, t/nd, n/(nd*nd)) * 2; // new lod coords basis
	
	const uint nnd = (1<<(depth+1)); // n of lods on each axis for depth-1
	uint off = 0; // offset for writing
	for (uint d = 0; d<depth; d++) { off += (1<<d*3); }
	uint j[8];
	j[0] = off + (1<<(depth*3)) + nlodcb.x +     (nlodcb.y    ) * nnd + (nlodcb.z    ) * nnd * nnd;
	j[1] = off + (1<<(depth*3)) + nlodcb.x + 1 + (nlodcb.y    ) * nnd + (nlodcb.z    ) * nnd * nnd;
	j[2] = off + (1<<(depth*3)) + nlodcb.x + 1 + (nlodcb.y + 1) * nnd + (nlodcb.z    ) * nnd * nnd;
	j[3] = off + (1<<(depth*3)) + nlodcb.x + 1 + (nlodcb.y + 1) * nnd + (nlodcb.z + 1) * nnd * nnd;
	j[4] = off + (1<<(depth*3)) + nlodcb.x + 1 + (nlodcb.y    ) * nnd + (nlodcb.z + 1) * nnd * nnd;
	j[5] = off + (1<<(depth*3)) + nlodcb.x +     (nlodcb.y + 1) * nnd + (nlodcb.z    ) * nnd * nnd;
	j[6] = off + (1<<(depth*3)) + nlodcb.x +     (nlodcb.y + 1) * nnd + (nlodcb.z + 1) * nnd * nnd;
	j[7] = off + (1<<(depth*3)) + nlodcb.x +     (nlodcb.y    ) * nnd + (nlodcb.z + 1) * nnd * nnd;
	//if (n == 0) printf("%u\n", j[5]);
	float qs = 0.0f, uxs = 0.0f, uys = 0.0f, uzs = 0.0f;
	for (uint i = 0; i<8; i++) {
		qs  += lods[j[i]*4+0];
		uxs += lods[j[i]*4+1];
		uys += lods[j[i]*4+2];
		uzs += lods[j[i]*4+3];
	}
	lods[(off+n)*4+0] = qs;
	lods[(off+n)*4+1] = uxs*0.125;
	lods[(off+n)*4+2] = uys*0.125;
	lods[(off+n)*4+3] = uzs*0.125;
}

__kernel void update_e_b_dynamic(const global float* E_stat, const global float* B_stat, global float* E_dyn, global float* B_dyn, const global float* Q, const global float* u, const global float* QU_lod, const global uchar* flags) {
	const uint n = get_global_id(0); // n = x+(y+z*Ny)*Nx
	if(n>=(uint)DEF_N||is_halo(n)) return; // don't execute update_e_b_dynamic() on halo
	const uchar flagsn = flags[n]; // cache flags[n] for multiple readings
	const uchar flagsn_bo=flagsn&TYPE_BO; // extract boundary and surface flags
	if(flagsn_bo==TYPE_S) return; // if cell is solid boundary or gas, just return

	const uint3 coord_n = coordinates(n); // Cell coordinate
	const float3 coord_nf = convert_float3(coord_n); // Cell coordinate as float vector

	const uint nd = (1<<DEF_LOD_DEPTH); // Number of lowest level lods on each axis
	const uint dsx = imax(DEF_NX / (1<<nd), 1);
	const uint dsy = imax(DEF_NY / (1<<nd), 1);
	const uint dsz = imax(DEF_NZ / (1<<nd), 1);

	const uint x_upper = imin((coord_n.x / dsx) * dsx + dsx, DEF_DX>1?DEF_NX-1:DEF_NX); // Do not read at halo offsets
	const uint y_upper = imin((coord_n.y / dsy) * dsy + dsy, DEF_DY>1?DEF_NY-1:DEF_NY);
	const uint z_upper = imin((coord_n.z / dsz) * dsz + dsz, DEF_DZ>1?DEF_NZ-1:DEF_NZ);

	float3 e = {0.0f, 0.0f, 0.0f}, b = {0.0f, 0.0f, 0.0f};
	
	/// Close distance - consider individual cells
	for(		uint x = imax((coord_n.x / dsx) * dsx, DEF_DX>1?1:0); x < x_upper; x++) {
		for(	uint y = imax((coord_n.y / dsy) * dsy, DEF_DY>1?1:0); y < y_upper; y++) {
			for(uint z = imax((coord_n.z / dsz) * dsz, DEF_DZ>1?1:0); z < z_upper; z++) {
				// _c vars describe surronding cells 
				const uint n_c = x + (y + z * DEF_NY) * DEF_NX;
				if (n == n_c) continue;
					
				const float q_c = Q[n_c]; // charge of nearby cell
				if (q_c == 0.0f) { continue; } // cells without charge have no influence
				const float3 v_c = {u[n_c], u[(ulong)n_c+DEF_N], u[(ulong)n_c+DEF_N*2ul]}; // velocity of nearby cell

				// precalculation for both fields
				const float3 vec_r = coord_nf - convert_float3(coordinates(n_c));
				const float3 pre_field = vec_r / cbmagnitude(vec_r);

				e += q_c * pre_field;			 // E imparted by nearby cell (Coulomb)
				b += q_c * cross(v_c, pre_field); // B imparted by nearby cell (Biot-Savart)
			}
		}
	}

	/// Medium distance - consider lowest level LODs in own domain
	const uint ndi = lod_index(n, DEF_LOD_DEPTH); // Own LOD index, needs to be skipped
	// Loop over all lowest level LODs
	for (uint d = imax(DEF_NUM_LOD_OWN - to_d(1<<DEF_LOD_DEPTH), 0); d < DEF_NUM_LOD_OWN; d++) {
		if (d == ndi) continue;
		const float3 d_c = lod_coordinates(d, DEF_LOD_DEPTH);
		const float  q_c =  QU_lod[(d*4)+0]; // charge of LOD
		const float3 v_c = {QU_lod[(d*4)+1], QU_lod[(d*4)+2], QU_lod[(d*4)+3]}; // velocity of LOD

		// precalculation for both fields
		const float3 vec_r = coord_nf - d_c;
		const float3 pre_field = vec_r / cbmagnitude(vec_r);

		e += q_c * pre_field; // E imparted by LOD (Coulomb)
		b += q_c * cross(v_c, pre_field); // B imparted by LOD (Biot-Savart)
	}

	/// Large distance - consider LODs of varying detail in foreign domains (synchronized over communicate_qu_lods)
	const uint3 coord_d = coordinates_sl(DEF_DI, DEF_DX, DEF_DY); // Own domain coordinate
	uint offset = DEF_NUM_LOD_OWN;
	for (uint d = 0; d < DEF_DX*DEF_DY*DEF_DZ; d++) { // Loop over every other domain
		if (d == DEF_DI) continue;
		const uint3 coord_fd = coordinates_sl(d, DEF_DX, DEF_DY); // coordinate of foreign domain
		const int3 domain_diff = {(int)coord_d.x - (int)coord_fd.x, (int)coord_d.y - (int)coord_fd.y, (int)coord_d.z - (int)coord_fd.z}; // Difference in domain coordinates
		const uint dist = imax(abs(domain_diff.x), imax(abs(domain_diff.y), abs(domain_diff.z)));
		const uint depth = imax(0, DEF_LOD_DEPTH - dist); // Depth of foreign domain LOD
		const uint n_lod_fd = to_d(1<<depth); // Number of lods in foreign domain

		for (int l = 0; l<n_lod_fd; l++) { // Loop over every LOD in foreign domain
			float3 lc = lod_coordinates(l, depth); // LOD coordinate
			lc.x -= (float)(domain_diff.x * DEF_NX);
			lc.y -= (float)(domain_diff.y * DEF_NY);
			lc.z -= (float)(domain_diff.z * DEF_NZ);
			const float  q_c =  QU_lod[(offset+l)*4+0]; // charge of LOD
			const float3 v_c = {QU_lod[(offset+l)*4+1], QU_lod[(offset+l)*4+2], QU_lod[(offset+l)*4+3]}; // velocity of LOD

			const float3 vec_r = coord_nf - lc;
			const float3 pre_field = vec_r / cbmagnitude(vec_r);

			e += q_c * pre_field; // E imparted by LOD (Coulomb)
			b += q_c * cross(v_c, pre_field); // B imparted by LOD (Biot-Savart)
		}
		offset += n_lod_fd;
	}

	// update buffers with static buffers and updated dynamic values
	E_dyn[n					] = E_stat[n				 ] + DEF_KE * e.x;
	E_dyn[(ulong)n+DEF_N	] = E_stat[(ulong)n+DEF_N	 ] + DEF_KE * e.y;
	E_dyn[(ulong)n+DEF_N*2ul] = E_stat[(ulong)n+DEF_N*2ul] + DEF_KE * e.z;

	B_dyn[n					] = B_stat[n				 ] + DEF_KMU * b.x;
	B_dyn[(ulong)n+DEF_N	] = B_stat[(ulong)n+DEF_N	 ] + DEF_KMU * b.y;
	B_dyn[(ulong)n+DEF_N*2ul] = B_stat[(ulong)n+DEF_N*2ul] + DEF_KMU * b.z;
} // update_e_b_dynamic()

__kernel void clear_qu_lod(global float* QU_lod) {
	// Clears own domain LODs for recomputation
	const uint n = get_global_id(0);
	if(n>DEF_NUM_LOD_OWN) return;
	QU_lod[(n * 4)+0] = 0.0f;
	QU_lod[(n * 4)+1] = 0.0f;
	QU_lod[(n * 4)+2] = 0.0f;
	QU_lod[(n * 4)+3] = 0.0f;
} // clear_qu_lod()
#endif

// Inter-Domain Transfer kernels
uint get_area(const uint direction) {
	const uint A[3] = { DEF_AX, DEF_AY, DEF_AZ };
	return A[direction];
}
// Return 1D index of cell to be transferred from id a and xyz direction (0, 1, 2) for pos and neg directions
uint index_extract_p(const uint a, const uint direction) {
	const uint3 coordinates[3] = { (uint3)(DEF_NX-2u, a%DEF_NY, a/DEF_NY), (uint3)(a/DEF_NZ, DEF_NY-2u, a%DEF_NZ), (uint3)(a%DEF_NX, a/DEF_NX, DEF_NZ-2u) };
	return index(coordinates[direction]);
}
uint index_extract_m(const uint a, const uint direction) {
	const uint3 coordinates[3] = { (uint3)(		  1u, a%DEF_NY, a/DEF_NY), (uint3)(a/DEF_NZ,		1u, a%DEF_NZ), (uint3)(a%DEF_NX, a/DEF_NX,		1u) };
	return index(coordinates[direction]);
}
uint index_insert_p(const uint a, const uint direction) {
	const uint3 coordinates[3] = { (uint3)(DEF_NX-1u, a%DEF_NY, a/DEF_NY), (uint3)(a/DEF_NZ, DEF_NY-1u, a%DEF_NZ), (uint3)(a%DEF_NX, a/DEF_NX, DEF_NZ-1u) };
	return index(coordinates[direction]);
}
uint index_insert_m(const uint a, const uint direction) {
	const uint3 coordinates[3] = { (uint3)(		  0u, a%DEF_NY, a/DEF_NY), (uint3)(a/DEF_NZ,		0u, a%DEF_NZ), (uint3)(a%DEF_NX, a/DEF_NX,		0u) };
	return index(coordinates[direction]);
}
// Returns an index for the transferred ddfs
uint index_transfer(const uint side_i) {
	const uchar index_transfer_data[2u*DEF_DIMENSIONS*DEF_TRANSFERS] = {
	#if defined(D2Q9)
		1,  5,  7, // xp
		2,  6,  8, // xm
		3,  5,  8, // yp
		4,  6,  7  // ym
	#elif defined(D3Q15)
		1,  7, 14,  9, 11, // xp
		2,  8, 13, 10, 12, // xm
		3,  7, 12,  9, 13, // yp
		4,  8, 11, 10, 14, // ym
		5,  7, 10, 11, 13, // zp
		6,  8,  9, 12, 14  // zm
	#elif defined(D3Q19)
		1,  7, 13,  9, 15, // xp
		2,  8, 14, 10, 16, // xm
		3,  7, 14, 11, 17, // yp
		4,  8, 13, 12, 18, // ym
		5,  9, 16, 11, 18, // zp
		6, 10, 15, 12, 17  // zm
	#elif defined(D3Q27)
		1,  7, 13,  9, 15, 19, 26, 21, 23, // xp
		2,  8, 14, 10, 16, 20, 25, 22, 24, // xm
		3,  7, 14, 11, 17, 19, 24, 21, 25, // yp
		4,  8, 13, 12, 18, 20, 23, 22, 26, // ym
		5,  9, 16, 11, 18, 19, 22, 23, 25, // zp
		6, 10, 15, 12, 17, 20, 21, 24, 26  // zm
	#endif // D3Q27
	};
	return (uint)index_transfer_data[side_i];
}

// Fi
void extract_fi(const uint a, const uint A, const uint n, const uint side, const ulong t, global fpxx_copy* transfer_buffer, const global fpxx_copy* fi) {
	uint j[DEF_VELOCITY_SET]; // neighbor indices
	neighbors(n, j); // calculate neighbor indices
	for(uint b=0u; b<DEF_TRANSFERS; b++) {
		const uint i = index_transfer(side*DEF_TRANSFERS+b);
		const ulong index = index_f(i%2u ? j[i] : n, t%2ul ? (i%2u ? i+1u : i-1u) : i); // Esoteric-Pull: standard store, or streaming part 1/2
		transfer_buffer[b*A+a] = fi[index]; // fpxx_copy allows direct copying without decompression+compression
	}
}
void insert_fi(const uint a, const uint A, const uint n, const uint side, const ulong t, const global fpxx_copy* transfer_buffer, global fpxx_copy* fi) {
	uint j[DEF_VELOCITY_SET]; // neighbor indices
	neighbors(n, j); // calculate neighbor indices
	for(uint b=0u; b<DEF_TRANSFERS; b++) {
		const uint i = index_transfer(side*DEF_TRANSFERS+b);
		const ulong index = index_f(i%2u ? n : j[i-1u], t%2ul ? i : (i%2u ? i+1u : i-1u)); // Esoteric-Pull: standard load, or streaming part 2/2
		fi[index] = transfer_buffer[b*A+a]; // fpxx_copy allows direct copying without decompression+compression
	}
}
kernel void transfer_extract_fi(const uint direction, const ulong t, global uchar* transfer_buffer_p, global uchar* transfer_buffer_m, const global fpxx_copy* fi) {
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of def_workgroup_size, so return here to avoid writing in unallocated memory space
	extract_fi(a, A, index_extract_p(a, direction), 2u*direction+0u, t, (global fpxx_copy*) transfer_buffer_p, fi);
	extract_fi(a, A, index_extract_m(a, direction), 2u*direction+1u, t, (global fpxx_copy*) transfer_buffer_m, fi);
}
kernel void transfer__insert_fi(const uint direction, const ulong t, const global uchar* transfer_buffer_p, const global uchar* transfer_buffer_m, global fpxx_copy* fi) {
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of def_workgroup_size, so return here to avoid writing in unallocated memory space
	insert_fi(a, A, index_insert_p(a, direction), 2u*direction+0u, t, (const global fpxx_copy*) transfer_buffer_p, fi);
	insert_fi(a, A, index_insert_m(a, direction), 2u*direction+1u, t, (const global fpxx_copy*) transfer_buffer_m, fi);
}
// Rho, u and flags (needed if graphics are active)
void extract_rho_u_flags(const uint a, const uint A, const uint n, global char* transfer_buffer, const global float* rho, const global float* u, const global uchar* flags) {
	((global float*)transfer_buffer)[	   a] = rho[			   n];
	((global float*)transfer_buffer)[	 A+a] = u[				   n];
	((global float*)transfer_buffer)[ 2u*A+a] = u[	  DEF_N+(ulong)n];
	((global float*)transfer_buffer)[ 3u*A+a] = u[2ul*DEF_N+(ulong)n];
	((global uchar*)transfer_buffer)[16u*A+a] = flags[			   n];
}
void insert_rho_u_flags(const uint a, const uint A, const uint n, const global char* transfer_buffer, global float* rho, global float* u, global uchar* flags) {
	rho[			   n] = ((const global float*)transfer_buffer)[		 a];
	u[				   n] = ((const global float*)transfer_buffer)[	   A+a];
	u[	  DEF_N+(ulong)n] = ((const global float*)transfer_buffer)[ 2u*A+a];
	u[2ul*DEF_N+(ulong)n] = ((const global float*)transfer_buffer)[ 3u*A+a];
	flags[			   n] = ((const global uchar*)transfer_buffer)[16u*A+a];
}
kernel void transfer_extract_rho_u_flags(const uint direction, const ulong t, global uchar* transfer_buffer_p, global uchar* transfer_buffer_m, const global float* rho, const global float* u, const global uchar* flags) {
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of def_workgroup_size, so return here to avoid writing in unallocated memory space
	extract_rho_u_flags(a, A, index_extract_p(a, direction), (global char*) transfer_buffer_p, rho, u, flags);
	extract_rho_u_flags(a, A, index_extract_m(a, direction), (global char*) transfer_buffer_m, rho, u, flags);
}
kernel void transfer__insert_rho_u_flags(const uint direction, const ulong t, const global uchar* transfer_buffer_p, const global uchar* transfer_buffer_m, global float* rho, global float* u, global uchar* flags) {
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of def_workgroup_size, so return here to avoid writing in unallocated memory space
	insert_rho_u_flags(a, A, index_insert_p(a, direction), (const global char*) transfer_buffer_p, rho, u, flags);
	insert_rho_u_flags(a, A, index_insert_m(a, direction), (const global char*) transfer_buffer_m, rho, u, flags);
}
// Qi (Charge ddfs) 
#ifdef MAGNETO_HYDRO
void extract_fqi(const uint a, const uint n, const uint side, const ulong t, global fpxx_copy* transfer_buffer, const global fpxx_copy* fqi) {
	uint j7[7u]; // neighbor indices
	neighbors_a(n, j7); // calculate neighbor indices
	const uint i = side+1u;
	const ulong index = index_f(i%2u ? j7[i] : n, t%2ul ? (i%2u ? i+1u : i-1u) : i); // Esoteric-Pull: standard store, or streaming part 1/2
	transfer_buffer[a] = fqi[index]; // fpxx_copy allows direct copying without decompression+compression
}
void insert_fqi(const uint a, const uint n, const uint side, const ulong t, const global fpxx_copy* transfer_buffer, global fpxx_copy* fqi) {
	uint j7[7u]; // neighbor indices
	neighbors_a(n, j7); // calculate neighbor indices
	const uint i = side+1u;
	const ulong index = index_f(i%2u ? n : j7[i-1u], t%2ul ? i : (i%2u ? i+1u : i-1u)); // Esoteric-Pull: standard load, or streaming part 2/2
	fqi[index] = transfer_buffer[a]; // fpxx_copy allows direct copying without decompression+compression
}
kernel void transfer_extract_fqi(const uint direction, const ulong t, global uchar* transfer_buffer_p, global uchar* transfer_buffer_m, const global fpxx_copy* fqi) {
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of def_workgroup_size, so return here to avoid writing in unallocated memory space
	extract_fqi(a, index_extract_p(a, direction), 2u*direction+0u, t, (global fpxx_copy*)transfer_buffer_p, fqi);
	extract_fqi(a, index_extract_m(a, direction), 2u*direction+1u, t, (global fpxx_copy*)transfer_buffer_m, fqi);
}
kernel void transfer__insert_fqi(const uint direction, const ulong t, const global uchar* transfer_buffer_p, const global uchar* transfer_buffer_m, global fpxx_copy* fqi) {
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of def_workgroup_size, so return here to avoid writing in unallocated memory space
	insert_fqi(a, index_insert_p(a, direction), 2u*direction+0u, t, (const global fpxx_copy*)transfer_buffer_p, fqi);
	insert_fqi(a, index_insert_m(a, direction), 2u*direction+1u, t, (const global fpxx_copy*)transfer_buffer_m, fqi);
}
#endif // MAGNETO_HYDRO

kernel void voxelize_mesh(const uint direction, global fpxx* fi, const global float* rho, global float* u, global uchar* flags, const ulong t, const uchar flag, const global float* p0, const global float* p1, const global float* p2, const global float* bbu
#ifdef MAGNETO_HYDRO
, const float mpc_x // magnetization per cell
, const float mpc_y
, const float mpc_z
, global float* B_dyn
#endif
){ // voxelize triangle mesh
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of def_workgroup_size, so return here to avoid writing in unallocated memory space
	const uint triangle_number = as_uint(bbu[0]);
	const float x0=bbu[ 1], y0=bbu[ 2], z0=bbu[ 3], x1=bbu[ 4], y1=bbu[ 5], z1=bbu[ 6];
	const float cx=bbu[ 7], cy=bbu[ 8], cz=bbu[ 9], ux=bbu[10], uy=bbu[11], uz=bbu[12], rx=bbu[13], ry=bbu[14], rz=bbu[15];
	const uint3 xyz = 
		direction==0u ? 
			(uint3)((uint)clamp((int)x0-DEF_OX, 0, (int)DEF_NX-1), a%DEF_NY, a/DEF_NY)
			: direction==1u ?
				(uint3)(a/DEF_NZ, (uint)clamp((int)y0-DEF_OY, 0, (int)DEF_NY-1), a%DEF_NZ)
				: (uint3)(a%DEF_NX, a/DEF_NX, (uint)clamp((int)z0-DEF_OZ, 0, (int)DEF_NZ-1));
	const float3 offset = (float3)(0.5f*(float)((int)DEF_NX+2*DEF_OX)-0.5f, 0.5f*(float)((int)DEF_NY+2*DEF_OY)-0.5f, 0.5f*(float)((int)DEF_NZ+2*DEF_OZ)-0.5f);
	const float3 r_origin = position(xyz)+offset;
	const float3 r_direction = (float3)((float)(direction==0u), (float)(direction==1u), (float)(direction==2u));
	uint intersections=0u, intersections_check=0u;
	ushort distances[64]; // allow up to 64 mesh intersections
	const bool condition = direction==0u ? r_origin.y<y0||r_origin.z<z0||r_origin.y>=y1||r_origin.z>=z1 : direction==1u ? r_origin.x<x0||r_origin.z<z0||r_origin.x>=x1||r_origin.z>=z1 : r_origin.x<x0||r_origin.y<y0||r_origin.x>=x1||r_origin.y>=y1;

	if(condition) return; // don't use local memory (~25% slower, but this also runs on old OpenCL 1.0 GPUs)
	for(uint i=0u; i<triangle_number; i++) {
		const uint tx=3u*i, ty=tx+1u, tz=ty+1u;
		const float3 p0i = (float3)(p0[tx], p0[ty], p0[tz]);
		const float3 p1i = (float3)(p1[tx], p1[ty], p1[tz]);
		const float3 p2i = (float3)(p2[tx], p2[ty], p2[tz]);
		const float3 u=p1i-p0i, v=p2i-p0i, w=r_origin-p0i, h=cross(r_direction, v), q=cross(w, u); // bidirectional ray-triangle intersection (Moeller-Trumbore algorithm)
		const float f=1.0f/dot(u, h), s=f*dot(w, h), t=f*dot(r_direction, q), d=f*dot(v, q);
		if(s>=0.0f&&s<1.0f&&t>=0.0f&&s+t<1.0f) { // ray-triangle intersection ahead or behind
			if(d>0.0f) { // ray-triangle intersection ahead
				if(intersections<64u&&d<65536.0f) distances[intersections] = (ushort)d; // store distance to intersection in array as ushort
				intersections++;
			} else { // ray-triangle intersection behind
				intersections_check++; // cast a second ray to check if starting point is really inside (error correction)
			}
		}
	}

	for(int i=1; i<(int)intersections; i++) { // insertion-sort distances
		ushort t = distances[i];
		int j = i-1;
		while(distances[j]>t&&j>=0) {
			distances[j+1] = distances[j];
			j--;
		}
		distances[j+1] = t;
	}

	bool inside = (intersections%2u)&&(intersections_check%2u);
	uint intersection = intersections%2u!=intersections_check%2u; // iterate through column, start with 0 regularly, start with 1 if forward and backward intersection count evenness differs (error correction)
	const uint h0 = direction==0u ? xyz.x : direction==1u ? xyz.y : xyz.z;
	const uint hmax = direction==0u ? (uint)clamp((int)x1-DEF_OX, 0, (int)DEF_NX) : direction==1u ? (uint)clamp((int)y1-DEF_OY, 0, (int)DEF_NY) : (uint)clamp((int)z1-DEF_OZ, 0, (int)DEF_NZ);
	const uint hmesh = h0+(uint)distances[min(intersections-1u, 63u)]; // clamp (intersections-1u) to prevent array out-of-bounds access
	for(uint h=h0; h<hmax; h++) {
		while(intersection<intersections&&h>h0+(uint)distances[min(intersection, 63u)]) { // clamp intersection to prevent array out-of-bounds access
			inside = !inside; // passed mesh intersection, so switch inside/outside state
			intersection++;
		}
		inside = inside&&(intersection<intersections&&h<hmesh); // point must be outside if there are no more ray-mesh intersections ahead (error correction)
		const ulong n = index((uint3)(direction==0u?h:xyz.x, direction==1u?h:xyz.y, direction==2u?h:xyz.z));
		uchar flagsn = flags[n];
		if(inside) {
			flagsn = (flagsn&~TYPE_BO)|flag;
			#ifdef MAGNETO_HYDRO
			if (flag&TYPE_M) { // magnetised solid, add magnetisation to b_dyn 
				B_dyn[          (ulong)n] = mpc_x;
				B_dyn[    DEF_N+(ulong)n] = mpc_y;
				B_dyn[2ul*DEF_N+(ulong)n] = mpc_z;
			} else if (flag&TYPE_F) { // charged solid, add charge to b_dyn 
				B_dyn[          (ulong)n] = mpc_x;
			}
			#endif // MAGNETO_HYDRO
			flags[n] = flagsn;
		}
	}
} // voxelize_mesh()

#ifdef MAGNETO_HYDRO
kernel void psi_from_mesh(const global uchar* flags, global float* psi, const global float* M) { // Psi field reuses the E_dyn buffer, M field reuses B_dyn buffer
	const uint n = get_global_id(0);
	const float3 c = convert_float3(coordinates_sl(n, DEF_NX+2, DEF_NY+2)); // coordinate in simulation, psi field is padded by 1 on each side
	float psic = 0.0f; // psi at cell

	for (int i = 0; i < DEF_N; i++) { // Iterate over every cell in simulation
		if (flags[i]&TYPE_M) { // cell is magnet
			const float3 cdiff = c - convert_float3(coordinates(i) + (uint3)(1, 1, 1));
			const float l = length(cdiff);
			if (!(l == 0.0f)) {
				const float3 mag = (float3)(M[i], M[i+DEF_N], M[i+DEF_N*2]);
				psic += dot(cdiff, mag) / cb(l);
			}
		}
	}

	psi[n] = psic / (4.0f * M_PI);
}

float3 nabla(const global float* v, const uint l0, const uint l1, const uint3 c) {
	const uint n = c.x + (c.y + c.z * l1) * l0;
	const uint yOff = l0;
	const uint zOff = l0 * l1;

	return (float3)(
		(v[n+1   ] - v[n-1   ]) / 2.0f,
		(v[n+yOff] - v[n-yOff]) / 2.0f,
		(v[n+zOff] - v[n-zOff]) / 2.0f
	);
}

kernel void static_b_from_mesh(const global uchar* flags, global float* B, const global float* psi) { // Psi field reuses the E_dyn buffer
	const uint n = get_global_id(0);
	if(n>=(uint)DEF_N||is_halo(n)) return; // don't execute static_b_from_mesh() on halo
	if((flags[n]&TYPE_S)==TYPE_S) return; // if cell is solid boundary or gas, just return

	const uint3 c = coordinates(n);
	const float3 Bc = -DEF_KMU0 * nabla(psi, DEF_NX+2, DEF_NY+2, c + (uint3)(1, 1, 1));

	B[(ulong)n         ] += Bc.x;
	B[(ulong)n+DEF_N   ] += Bc.y;
	B[(ulong)n+DEF_N*2u] += Bc.z;
}

kernel void static_e_from_mesh(const global uchar* flags, global float* E, const global float* C) { // Psi field reuses the E_dyn buffer
	const uint n = get_global_id(0);
	if(n>=(uint)DEF_N||is_halo(n)) return; // don't execute static_e_from_mesh() on halo
	if((flags[n]&TYPE_S)==TYPE_S) return; // if cell is solid boundary or gas, just return

	const float3 c = convert_float3(coordinates(n));
	float3 Ec = (float3)(0.0f, 0.0f, 0.0f);

	for (int i = 0; i < DEF_N; i++) { // Iterate over every cell in simulation
		if (flags[i]&TYPE_F) { // cell is charge
			const float3 cdiff = c - convert_float3(coordinates(i));
			const float l = length(cdiff);
			if (!(l == 0.0f)) {
				const float3 charge = C[i];
				Ec += cdiff * charge / cb(l);
			}
		}
	}

	E[(ulong)n         ] += Ec.x;
	E[(ulong)n+DEF_N   ] += Ec.y;
	E[(ulong)n+DEF_N*2u] += Ec.z;
}
#endif // MAGNETO_HYDRO