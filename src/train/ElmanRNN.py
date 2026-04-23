"""
ElmanRNN.py

PyTorch implementation of the recurrent network used in this project.

This module defines a lightweight Elman-style RNN wrapper around ``torch.nn.RNN``.
It supports the standard PyTorch nonlinearities (``tanh`` and ``relu``) and also
a manually implemented ``linear`` hidden update for experiments that require an
identity hidden-state nonlinearity.

Author: Margot Wagner
Date: 2026-04-23

Notes
-----
- Input tensors are expected to have shape ``(batch, time, input_dim)``.
- Initial hidden states may be provided either as ``(1, batch, hidden_dim)``
  or ``(batch, hidden_dim)``.
- The output layer applies a softmax across the feature dimension at each time
  step, returning class-like output probabilities together with the hidden-state
  trajectory.
"""

import torch
import torch.nn as nn


class ElmanRNN(nn.Module):
    """
    Single-layer Elman RNN with an output projection.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step.
    hidden_dim : int
        Number of recurrent hidden units.
    output_dim : int
        Number of output features per time step.
    rnn_act : {"tanh", "relu", "linear"}, default="tanh"
        Hidden-state nonlinearity. ``"tanh"`` and ``"relu"`` use the built-in
        PyTorch RNN implementation. ``"linear"`` reuses the same learned weight
        tensors but applies the recurrence manually without a hidden nonlinearity.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        rnn_act: str = "tanh",
    ) -> None:
        super().__init__()

        if rnn_act not in {"tanh", "relu", "linear"}:
            raise ValueError(
                f"Unsupported rnn_act={rnn_act!r}. Expected 'tanh', 'relu', or 'linear'."
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = 1
        self.rnn_act = rnn_act

        # torch.nn.RNN only supports tanh and relu internally. For the linear
        # case we still construct the RNN module so we can reuse its parameter
        # tensors, but we apply the recurrence manually in forward().
        nonlinearity = "relu" if rnn_act == "relu" else "tanh"
        self.rnn = nn.RNN(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            batch_first=True,
            nonlinearity=nonlinearity,
        )

        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.act_output = nn.Softmax(dim=2)  # activation function

    def forward(self, x, h0):
        """
        Run the network forward.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape ``(batch, time, input_dim)``.
        h0 : torch.Tensor
            Initial hidden state with shape ``(1, batch, hidden_dim)`` or
            ``(batch, hidden_dim)``.

        Returns
        -------
        out : torch.Tensor
            Softmax-normalized outputs of shape ``(batch, time, output_dim)``.
        z : torch.Tensor
            Hidden-state trajectory of shape ``(batch, time, hidden_dim)``.
        """
        if self.rnn_act == "linear":
            z = self._forward_linear_hidden(x, h0)
        else:
            # Standard tanh/relu recurrence handled by PyTorch
            z, _ = self.rnn(x, h0)

        out = self.act_output(self.linear(z))
        return out, z

    def _forward_linear_hidden(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Apply the recurrent update with an identity hidden nonlinearity.

        This path reuses the input-to-hidden and hidden-to-hidden parameters from
        the underlying ``nn.RNN`` module so that initialization and checkpoint
        handling remain consistent with the nonlinear cases.
        """
        _, seq_len, _ = x.shape
        h = h0[0] if h0.ndim == 3 else h0  # (batch, hidden_dim)

        weight_ih = self.rnn.weight_ih_l0  # (hidden_dim, input_dim)
        weight_hh = self.rnn.weight_hh_l0  # (hidden_dim, hidden_dim)
        bias_ih = getattr(self.rnn, "bias_ih_l0", None)
        bias_hh = getattr(self.rnn, "bias_hh_l0", None)

        hidden_sequence = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_dim)
            h = x_t @ weight_ih.t() + h @ weight_hh.t()

            if bias_ih is not None:
                h = h + bias_ih
            if bias_hh is not None:
                h = h + bias_hh

            hidden_sequence.append(h.unsqueeze(1))  # (batch, 1, hidden_dim)

        return torch.cat(hidden_sequence, dim=1)  # (batch, time, hidden_dim)
