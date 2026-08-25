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

    inp, hx, params = _make_case(
        4,
        2,
        16,
        16,
        torch.float32,
        num_layers=1,
        bidirectional=True,
    )
    for direction in range(2):
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
            1,
            0.0,
            True,
            True,
            False,
        )
        start = direction * 16
        (ref_output[..., start : start + 16].sum() + ref_hidden[direction].sum()).backward()

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
            1,
            0.0,
            True,
            True,
            False,
        )
        (output[..., start : start + 16].sum() + hidden[direction].sum()).backward()
        print(
            "DIRECTION",
            direction,
            (gems_input.grad - ref_input.grad).abs().max().item(),
            (gems_hx.grad - ref_hx.grad).abs().max().item(),
            [
                (actual.grad - expected.grad).abs().max().item()
                if actual.grad is not None and expected.grad is not None
                else None
                for actual, expected in zip(gems_params, ref_params)
            ],
        )

    for seq_len in (1, 2, 4):
        inp, hx, params = _make_case(
            seq_len,
            2,
            16,
            16,
            torch.float32,
            num_layers=1,
            bidirectional=True,
        )
        grads = []
        for implementation in (torch.rnn_tanh, rnn_tanh):
            current_input = inp.detach().clone().requires_grad_(True)
            current_hx = hx.detach().clone().requires_grad_(True)
            current_params = tuple(
                param.detach().clone().requires_grad_(True) for param in params
            )
            output, hidden = implementation(
                current_input,
                current_hx,
                current_params,
                True,
                1,
                0.0,
                True,
                True,
                False,
            )
            (output[..., 16:].sum() + hidden[1].sum()).backward()
            grads.append(
                (
                    current_input.grad,
                    current_hx.grad,
                    tuple(param.grad for param in current_params),
                )
            )
        print(
            "REVERSE_LENGTH",
            seq_len,
            (grads[1][0] - grads[0][0]).abs().max().item(),
            (grads[1][1] - grads[0][1]).abs().max().item(),
            max(
                (actual - expected).abs().max().item()
                for actual, expected in zip(grads[1][2][4:], grads[0][2][4:])
            ),
        )
