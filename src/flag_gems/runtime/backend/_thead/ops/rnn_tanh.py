# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from flag_gems.ops.rnn_tanh import _rnn_tanh_impl
from flag_gems.ops.rnn_tanh import rnn_tanh_data as _generic_rnn_tanh_data

logger = logging.getLogger(__name__)


def rnn_tanh(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    """T-Head RNN-tanh using its validated NVIDIA-style Triton selector."""
    logger.debug("GEMS T-HEAD RNN_TANH")
    return _rnn_tanh_impl(
        input,
        hx,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        prefer_persistent_dot=True,
    )


def rnn_tanh_data(
    data,
    batch_sizes,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
):
    """T-Head packed RNN-tanh using the shared pure-Triton implementation."""
    logger.debug("GEMS T-HEAD RNN_TANH DATA")
    return _generic_rnn_tanh_data(
        data,
        batch_sizes,
        hx,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
    )
