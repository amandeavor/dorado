#pragma once

namespace dorado::nn {

// TODO: These should really be part of Koi
bool koi_can_use_cutlass(/* current device */);
bool koi_can_use_cutlass(int device_id);
bool koi_can_use_quantised_lstm(/* current device */);
bool koi_can_run_flstm(/* current device */);
bool koi_can_run_flstm(int device_id);
bool koi_can_run_tx_vcs(int device_id);

}  // namespace dorado::nn
