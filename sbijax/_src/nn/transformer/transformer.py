from flax import nnx
from .encoder import EncoderBlock, EncoderDecoderBlock
from .embedding import Embedding

class Transformer(nnx.Module):
    """Transformer with encoder-decoder architecture
    which encodes observations and decodes a vector field
    for flow matching.
    """

    def __init__(
        self,
        config,
        context_value_dim,
        n_context_labels,
        context_index_dim,
        theta_value_dim,
        n_theta_labels,
        theta_index_dim,
        rngs=nnx.Rngs(0)
        ):
        self.context_embedding = Embedding(
            context_value_dim,
            n_context_labels,
            config['label_dim'],
            context_index_dim,
            config['index_out_dim'],
            config['latent_dim'],
            rngs=rngs
        )

        @nnx.split_rngs(splits=config['n_encoder'])
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_encoder(rngs):
            return EncoderBlock(
                config['n_heads'],
                config['latent_dim'],
                config['n_ff'],
                config['dropout'],
                config['activation'],
                rngs=rngs
            )

        self.encoder = create_encoder(rngs)
        self.n_encoder = config['n_encoder']

        self.pos_embedding = Embedding(
            theta_value_dim,
            n_theta_labels,
            config['label_dim'],
            theta_index_dim,
            config['index_out_dim'],
            config['latent_dim'],
            rngs=rngs
        )

        @nnx.split_rngs(splits=config['n_decoder'])
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_decoder(rngs):
            return EncoderDecoderBlock(
                config['n_heads'],
                config['latent_dim'],
                config['n_ff'],
                config['dropout'],
                config['activation'],
                rngs=rngs
            )

        self.decoder = create_decoder(rngs)
        self.n_decoder = config['n_decoder']
        self.linear = nnx.Linear(config['latent_dim'], 1, rngs=rngs)

    def __call__(
        self,
        context,
        context_label,
        context_index,
        theta,
        theta_label,
        theta_index,
        time
        ):
        encoded = self.encode(
            context,
            context_label,
            context_index,
            time
        )
        decoded = self.decode(
            theta,
            theta_label,
            theta_index,
            encoded,
            time
        )
        return decoded

    def encode(
        self,
        context,
        context_label,
        context_index,
        time
        ):
        x = self.context_embedding(
            context,
            context_label,
            context_index,
            time
        )
        @nnx.split_rngs(splits=self.n_encoder)
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def forward(x, model):
            x = model(x)
            return x

        return forward(x, self.encoder)

    def decode(
        self,
        pos,
        pos_label,
        pos_index,
        encoded,
        time
        ):
        x = self.pos_embedding(
            pos,
            pos_label,
            pos_index,
            time
        )

        @nnx.split_rngs(splits=self.n_decoder)
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def forward(x, model):
            x = model(x, encoded)
            return x
        x = forward(x, self.decoder)
        return self.linear(x)
