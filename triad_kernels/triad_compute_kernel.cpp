// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/eltwise_binary.h"

#include <cstdint>
#include "debug/dprint.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_in1 = tt::CB::c_in1;
    constexpr auto cb_in2 = tt::CB::c_in2;
    constexpr auto cb_out0 = tt::CB::c_out0;
    constexpr auto cb_int = tt::CB::c_intermed0;

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        binary_op_init_common(cb_in1, cb_in2, cb_int);

        DPRINT_MATH(DPRINT << "1. pre mul acquire" << ENDL());

        acquire_dst(tt::DstMode::Full);

        cb_wait_front(cb_in1, per_core_block_size);
        cb_wait_front(cb_in2, per_core_block_size);

        DPRINT_MATH(DPRINT << "2. post mul acquire, pre mul op" << ENDL());

        mul_tiles_init();
        mul_tiles(cb_in1, cb_in2, 0, 0, 0);
        cb_reserve_back(cb_int, per_core_block_size);

        pack_tile(0, cb_int);

        cb_push_back(cb_int, 1);

        cb_pop_front(cb_in1, 1);
        cb_pop_front(cb_in2, 1);

        release_dst(tt::DstMode::Full);

        DPRINT_MATH(DPRINT << "3. post mul release, pre add acquire" << ENDL());

        acquire_dst(tt::DstMode::Full);

        binary_op_init_common(cb_in0, cb_int, cb_out0);

        cb_wait_front(cb_in0, per_core_block_size);
        cb_wait_front(cb_int, per_core_block_size);

        DPRINT_MATH(DPRINT << "4. post add acquire, pre add op" << ENDL());

        add_tiles_init();
        add_tiles(cb_in0, cb_int, 0, 0, 0);

        cb_reserve_back(cb_out0, per_core_block_size);

        pack_tile(0, cb_out0);

        cb_push_back(cb_out0, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_int, 1);

        release_dst(tt::DstMode::Full);

        DPRINT_MATH(DPRINT << "5. post add release" << ENDL());
    }
}
} 

