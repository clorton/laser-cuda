// Helper for XOROSHIRO128+ PRNG
__device__ inline uint64_t rotl(uint64_t x, uint32_t k) {
    return (x << k) | (x >> (64 - k));
}

__device__ uint64_t get_next(uint64_t *states, uint32_t index) {

    uint64_t s0 = states[index*2];
    uint64_t s1 = states[index*2+1];
    uint64_t result = s0 + s1;

    s1 ^= s0;
    s0 = rotl(s0, 55) ^ s1 ^ (s1 << 14);
    s1 = rotl(s1, 36);

    states[index*2] = s0;
    states[index*2+1] = s1;

    return result;
}

#define FACTOR (double(1.0) / 9007199254740992)

__device__ inline float uint64_to_unit_float32(uint64_t x) {
    return float((x >> 11) * FACTOR);
}

#define TWO_PI  (float(2.0 * 3.14159265))

__device__ float draw_normal(uint64_t s0, uint64_t s1, float mean, float std) {

    float u1 = uint64_to_unit_float32(s0);
    float u2 = uint64_to_unit_float32(s1);
    float z0 = sqrt(-2 * log(u1)) * cos(TWO_PI * u2);
    return (z0 * std + mean);
}

__global__ void infect(uint32_t count, uint32_t *indices, uint64_t *prng_states, uint8_t *susceptibility, uint8_t *itimer, float inf_mean, float inf_std) {
    int ithread = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ uint64_t prns[];
    if (ithread < count) {
        if (ithread == 0) {
            uint64_t *pprns = prns;
            for (int j = 0; j < count; ++j) {
                *pprns++ = get_next(prng_states, 0);
                *pprns++ = get_next(prng_states, 0);
            }
        }
        __syncthreads();    // Wait here until all PRNS have been generated.
        uint32_t iagent = indices[ithread];
        uint32_t iS0 = ithread * 2;
        uint32_t iS1 = iS0 + 1;
        susceptibility[iagent] = 0;
        itimer[iagent] = uint8_t(__roundf(draw_normal(prns[iS0], prns[iS1], inf_mean, inf_std)));
    }
}

__global__ void infection_update(uint32_t count, uint8_t *itimer) {
    int iagent = blockIdx.x * blockDim.x + threadIdx.x;
    if (iagent < count) {
        if (itimer[iagent] > 0) {
            itimer[iagent] -= 1;
        }
    }
}

__global__ void exposure_update(
    uint32_t count,
    uint8_t *etimer,
    uint64_t *prng_states,
    uint32_t stride,
    uint8_t *itimer,
    float inf_mean,
    float inf_std)
    {
    int iagent = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ uint64_t prns[];
    if (iagent < count) {
        /*
        Use the initial PRN element to store the infection status.
        Assume no one is getting infected.
         */
        uint32_t generatorS0 = (threadIdx.x - threadIdx.x % stride) * 2;

        bool isGenerator = (threadIdx.x % stride == 0);
        if (isGenerator) {
            prns[generatorS0] = 0;  // Each thread gets two PRNs.
        }
        __syncthreads();            // Wait here until PRN[iprng*2] is set to 0 by the prng thread.

        /*
        Decrement the exposure counter, if non-zero.
        If it hits zero, note in PRN[0] that at least one agent is now infected.
        Remember this in `infected`.
        */
        bool infectious = false;
        if (etimer[iagent] > 0) {
            uint8_t newtimer = etimer[iagent] - 1;
            etimer[iagent] = newtimer;
            if (newtimer == 0) {
                prns[generatorS0] = 1;  // Each thread gets two PRNs.
                infectious = true;
            }
        }
        __syncthreads();    // Wait here until everyone has a chance to set PRN[iprng*2].

        if (isGenerator && (prns[generatorS0] != 1)) {
            /*
            Each thread in `stride` gets two PRNs in case it needs to do a normal
            distribution draw for the duration of infectiousness.
            */
            uint64_t *pprns = prns + generatorS0;
            uint32_t istate = iagent / stride;  // E.g. 0, 0, 0, 0, 1, 1, 1, 1, ... based on `stride`
            for (int j = 0; j < stride; ++j) {
                *pprns++ = get_next(prng_states, istate);
                *pprns++ = get_next(prng_states, istate);
            }
        }
        __syncthreads();    // Wait here until all PRNS have been generated.

        if (infectious) {
            uint32_t myprnS0 = threadIdx.x * 2;
            uint32_t myprnS1 = myprnS0 + 1;
            itimer[iagent] = uint8_t(__roundf(draw_normal(prns[myprnS0], prns[myprnS1], inf_mean, inf_std)));
        }
    }
}

__global__ void transmission_update(
    uint32_t count,
    uint64_t *prng_states,
    uint32_t stride,
    float force,
    uint8_t *susceptibility,
    uint8_t *etimer,
    float exp_mean,
    float exp_std)
    {
    int iagent = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ uint64_t prns[];
    if (iagent < count) {
        uint32_t generatorS0 = (threadIdx.x - threadIdx.x % stride) * 2;
        uint32_t generatorS1 = generatorS0 + 1;
        bool isGenerator = (threadIdx.x % stride == 0);
        if (isGenerator) {
            uint32_t istate = iagent / stride;
            uint64_t *pprns = prns + generatorS0;
            for (int j = 0; j < stride; ++j) {
                *pprns = get_next(prng_states, istate);
                pprns += 2;
            }
            prns[generatorS1] = 0;  // Each thread gets two PRNs.
        }
        __syncthreads();    // Wait here until all PRNS have been generated.

        bool exposed = false;
        uint32_t myprnS0 = threadIdx.x * 2;
        float uniform_draw = uint64_to_unit_float32(prns[myprnS0]);
        if (uniform_draw < (force * susceptibility[iagent])) {
            susceptibility[iagent] = 0;
            prns[generatorS1] = 1;
            exposed = true;
        }
        __syncthreads();    // Wait here until everyone has a chance to set PRNS[generatorS1].

        if (isGenerator && (prns[generatorS1] != 0)) {
            /*
            Each thread in `stride` gets two PRNs in case it needs to do a normal
            distribution draw for the duration of exposure.
            */
            uint32_t istate = iagent / stride;
            uint64_t *pprns = prns + generatorS0;
            for (int j = 0; j < stride; ++j) {
                *pprns++ = get_next(prng_states, istate);
                *pprns++ = get_next(prng_states, istate);
            }
        }
        __syncthreads();    // Wait here until all PRNS have been generated.

        if (exposed) {
            uint32_t myprnS1 = myprnS0 + 1;
            etimer[iagent] = uint8_t(round(draw_normal(prns[myprnS0], prns[myprnS1], exp_mean, exp_std)));
        }
    }
}