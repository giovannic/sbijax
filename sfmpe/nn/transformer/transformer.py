from flax import nnx
from .encoder import EncoderBlock, DecoderBlock
from .embedding import Embedding

class Transformer(nnx.Module):
    """Transformer with encoder-decoder architecture
    which encodes context and decodes a vector field
    for flow matching.
    """

    def __init__(
        self,
        config,
        value_dim,
        n_labels,
        index_dim,
        rngs=nnx.Rngs(0)
        ):
        self.embedding = Embedding(
            value_dim,
            n_labels,
            config['label_dim'],
            index_dim,
            config['index_out_dim'],
            config['latent_dim'],
            rngs=rngs
        )

        @nnx.split_rngs(splits=config['n_encoder'])
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_encoder(rngs):
            return EncoderBlock(
                dim=config['latent_dim'],
                n_ff=config['n_ff'],
                dropout=config['dropout'],
                activation=config['activation'],
                n_heads=config['n_heads'],
                rngs=rngs
            )

        self.encoder = create_encoder(rngs)
        self.n_encoder = config['n_encoder']

        @nnx.split_rngs(splits=config['n_decoder'])
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_decoder(rngs):
            return DecoderBlock(
                dim=config['latent_dim'],
                n_ff=config['n_ff'],
                dropout=config['dropout'],
                activation=config['activation'],
                n_heads=config['n_heads'],
                rngs=rngs
            )

        self.decoder = create_decoder(rngs)
        self.n_decoder = config['n_decoder']
        self.linear = nnx.Linear(
            config['latent_dim'],
            value_dim,
            rngs=rngs
        )

    def __call__(
        self,
        context,
        context_label,
        context_index,
        context_mask,
        theta,
        theta_label,
        theta_index,
        theta_mask,
        cross_mask,
        time,
        ):
        encoded = self.encode(
            context,
            context_label,
            context_index,
            context_mask,
            time
        )
        decoded = self.decode(
            theta,
            theta_label,
            theta_index,
            theta_mask,
            encoded,
            cross_mask,
            time,
        )
        return decoded

    def encode(
        self,
        context,
        context_label,
        context_index,
        context_mask,
        time
        ):
        x = self.embedding(
            context,
            context_label,
            context_index,
            time,
            is_context=True
        )
        @nnx.split_rngs(splits=self.n_encoder)
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def forward(x, model):
            x = model(x, mask=context_mask)
            return x

        return forward(x, self.encoder)

    def decode(
        self,
        pos,
        pos_label,
        pos_index,
        pos_mask,
        encoded,
        cross_mask,
        time,
        ):
        x = self.embedding(
            pos,
            pos_label,
            pos_index,
            time
        )
        @nnx.split_rngs(splits=self.n_encoder)
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def enc(x, model):
            x = model(x, mask=pos_mask)
            return x

        @nnx.split_rngs(splits=self.n_decoder)
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def dec(x, decoder):
            x = decoder(x, encoded, mask=cross_mask)
            return x

        x = enc(x, self.encoder)
        x = dec(x, self.decoder)
        return self.linear(x)
