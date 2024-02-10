import mlx.core as mx
import mlx.nn as nn


class PromptTuning(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        model: nn.Module
    ):
        super().__init__()

        # Regular linear layer weights
        self.model = model
        self.prompts = nn.Embedding(num_tokens, self.model.model.args.hidden_size)
        self.prompt_size = num_tokens

    def __call__(self, inputs, cache=None):
        # dtype = self.prompts.weight.dtype
        # if isinstance(self.prompts, nn.QuantizedLinear):
        #     dtype = self.prompts.scales.dtype
        prompt_embeds = mx.reshape(self.prompts(mx.arange(self.prompt_size)),
                                   (1, self.prompt_size, -1))
        prompt_embeds = mx.repeat(prompt_embeds, inputs.shape[0], axis=0)
        input_embeds = self.model.model.embed_tokens(inputs)
        input_embeds = mx.concatenate([prompt_embeds, input_embeds], axis=1)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(input_embeds.shape[1])
        mask[:, :10] = 0  # TODO: Or is this [:10, :]?
        mask = mask.astype(input_embeds.dtype)

        output, cache = self.model(inputs, input_embeds, mask, cache)
        output = output[:, self.prompt_size:, :]
        return output, cache
