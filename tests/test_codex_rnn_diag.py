import torch

from flag_gems.ops.rnn_tanh import rnn_tanh

from .test_rnn_tanh import _make_case


def test_codex_rnn_diag():
    for layers, bidirectional in ((1, False), (1, True), (2, False), (2, True)):
        inp, hx, params = _make_case(
            4,
            2,
            16,
            16,
            torch.float32,
            num_layers=layers,
            bidirectional=bidirectional,
        )
        ref_input = inp.detach().clone().requires_grad_(True)
        ref_hx = hx.detach().clone().requires_grad_(True)
        ref_params = tuple(
            param.detach().clone().requires_grad_(True) for param in params
        )
        ref_output, ref_hidden = torch.rnn_tanh(
            ref_input,
            ref_hx,
            ref_params,
            True,
            layers,
            0.0,
            True,
            bidirectional,
            False,
        )
        (ref_output.sum() + ref_hidden.sum()).backward()

        gems_input = inp.detach().clone().requires_grad_(True)
        gems_hx = hx.detach().clone().requires_grad_(True)
        gems_params = tuple(
            param.detach().clone().requires_grad_(True) for param in params
        )
        output, hidden = rnn_tanh(
            gems_input,
            gems_hx,
            gems_params,
            True,
            layers,
            0.0,
            True,
            bidirectional,
            False,
        )
        (output.sum() + hidden.sum()).backward()
        param_error = max(
            (actual.grad - expected.grad).abs().max().item()
            for actual, expected in zip(gems_params, ref_params)
        )
        print(
            "DIAG",
            layers,
            bidirectional,
            (gems_input.grad - ref_input.grad).abs().max().item(),
            (gems_hx.grad - ref_hx.grad).abs().max().item(),
            param_error,
        )
