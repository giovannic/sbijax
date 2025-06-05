# Implement an SIR model in JAX using an ODE
# and perform inference using (s)FMPE
# calculate validation statistics

from jax import (
    numpy as jnp,
    random as jr,
    vmap,
    tree
)
import tensorflow_probability.substrates.jax as tfp
from jax.experimental.ode import odeint

from sfmpe.fmpe import SFMPE
from sfmpe.nn.transformer.transformer import Transformer
from sfmpe.nn.make_continuous_flow import CNF
from sfmpe.util.dataloader import (
    flatten_structured,
    flat_as_batch_iterators,
    pad_multidim_event,
    combine_data
)

from flax import nnx

tfd = tfp.distributions
tfb = tfp.bijectors

def prior_fn(n_z):
    return tfd.JointDistributionNamed(dict(
        beta_0 = tfd.uniform([1e-3], [.5]), #type: ignore
        amp = lambda beta: tfd.Uniform(jnp.zeros((n_z,)), beta), #type: ignore
        phi = tfd.Uniform(jnp.zeros((n_z,)), jnp.pi), #type: ignore
        t_season = tfd.Uniform(
            jnp.full((n_z,), 365 * 2 - 100.), #type: ignore
            jnp.full((n_z,), 365 * 2 + 100.), #type: ignore
        ),
    ))

# ode function for SIR model
def dy_dt(t, state, *args):
    s, i, _, _, _ = state
    (
        beta_0,
        amp,
        phi,
        t_season
    ) = args

    gamma = 1./14. # recovery rate

    beta = beta_0 * (1 + amp * jnp.sin(2 * jnp.pi * t / t_season - phi))
    return jnp.array([
        -beta * s * i, # susceptible
        beta * s * i - gamma * i, # infected
        gamma * i, # recovered
    ])

def simulator_fn(seed, theta, **kwargs):
    beta_0 = theta['beta_0']
    amp = theta['amp']
    phi = theta['phi']
    t_season = theta['t_season']
    pop = kwargs['pop']
    inf_0 = kwargs['inf_0']

    # solve ode
    y0 = jnp.array([pop - inf_0, inf_0, 0., 0., 0.])
    y = vmap(
        odeint,
        in_axes=(
            None, #dy_dt
            None, #y0
            0, #beta0
            0, #amp
            0, #phi
            0 #t_season
        )
    )(
        dy_dt,
        y0,
        beta_0,
        amp,
        phi,
        t_season
    )
    inf = y[:, 1]
    inc = jnp.diff(inf)

    # simulate observed incidence according to a poisson distribution
    return {
        'obs_inc': tfd.Poisson(inc).sample(seed=seed)
    }

def sfmpe_inference(
    key,
    sample_size,
    obs,
    post_samples,
    n_iter,
    theta_cal,
    x_cal
    ):

    n_rounds = 2
    n_simulations = 100
    n_epochs = 100
    n_z = 10

    key, sample_key, fit_key, post_key = jr.split(key, 4)
    rngs = nnx.Rngs(fit_key)
    config = {
        'latent_dim': 12,
        'label_dim': 2,
        'index_out_dim': 4,
        'n_encoder': 1,
        'n_decoder': 1,
        'n_heads': 2,
        'n_ff': 2,
        'dropout': .1,
        'activation': nnx.relu,
    }
    nn = Transformer(
        config,
        value_dim=1,
        n_labels=4,
        index_dim=1,
        rngs=rngs
    )
    model = CNF(transform=nn)

    estim = SFMPE(model)

    independence = {
        'cross': [
            ('beta_0', 'amp'),
            ('beta_0', 'phi'),
            ('beta_0', 't_season')
        ],
    }

    flat_data, data_slices = (None, {})
    train_data = None
    for _ in range(n_rounds):
        # Fit p(z|theta, y)
        theta_key, obs_key, key = jr.split(key, 3)
        if flat_data is not None and data_slices:
            choice_key, theta_key = jr.split(theta_key)
            flat_y_obs, data_slices = flatten_structured(
                {
                    'theta': data['theta'], #type: ignore
                    'y': obs
                },
                independence=independence
            )
            theta_samples = estim.sample_structured_posterior(
                theta_key,
                flat_y_obs['data']['y'],
                flat_data['labels'], #type: ignore
                data_slices['theta'], #type: ignore
                masks=flat_data['masks'], #type: ignore
                n_samples=n_simulations
            )
            choice_key = jr.split(choice_key, 3)
            theta_samples = {
                'beta_0': theta_samples['beta_0'],
                'amp': jr.choice(choice_key[0], theta_samples['amp'], shape=(1,), axis=1),
                'phi': jr.choice(choice_key[1], theta_samples['phi'], shape=(1,), axis=1),
                't_season': jr.choice(choice_key[2], theta_samples['t_season'], shape=(1,), axis=1)

            }
        else:
            theta_samples = prior_fn(1).sample(seed=theta_key, sample_shape=(n_simulations,))

        y_samples= simulator_fn(obs_key, theta_samples)

        z_data = {
            'theta': {
                'amp': theta_samples['amp'], #type: ignore
                'phi': theta_samples['phi'], #type: ignore
                't_season': theta_samples['t_season'], #type: ignore
            },
            'y': {
                'beta_0': theta_samples['beta_0'], #type: ignore
                'obs': y_samples['obs']
            }
        }

        z_flat, _ = flatten_structured(
            z_data,
            independence=independence
        )

        if train_data is None:
            train_data = z_flat
        else:
            train_data = combine_data(train_data, z_flat)

        itr_key, key = jr.split(key)
        train_iter, val_iter = flat_as_batch_iterators(
            itr_key,
            train_data
        )

        fit_key, key = jr.split(key)

        estim.fit(
            fit_key,
            train_iter,
            val_iter,
            n_iter=n_epochs
        )

        # simulate from p(z_vec|theta, y_vec)
        z_sim = z_data.copy()
        sim_key, key = jr.split(key)
        z_sim['y']['obs'] = vmap(
            lambda k: jr.choice(k, z_data['y']['obs'], shape=(n_z,)) # type: ignore
        )(jr.split(sim_key, n_simulations))

        # pad z to n_z
        z_sim['theta']['z'] = pad_multidim_event(
            z_sim['theta']['z'],
            1,
            (n_z,)
        )

        (
            flat_z_sim,
            z_sim_slices
        ) = flatten_structured(
            z_sim,
            independence=independence
        )

        sample_key, key = jr.split(key)

        z_vec = vmap(
            lambda key, obs: tree.map(
                lambda leaf: leaf[0], #TODO: clean up
                estim.sample_structured_posterior(
                    key,
                    jnp.expand_dims(obs, 0),
                    flat_z_sim['labels'],
                    z_sim_slices['theta'],
                    masks=flat_z_sim['masks'],
                    n_samples=1
                )
            )
        )(jr.split(sample_key, n_simulations), flat_z_sim['data']['y'])

        # fit p(theta,z_vec|y_vec)
        data = {
            'theta': {
                'beta_0': z_sim['y']['beta_0'],
                'amp': z_vec['amp'],
                'phi': z_vec['phi'],
                't_season': z_vec['t_season'],
            },
            'y': {
                'obs': z_sim['y']['obs']
            }
        }

        flat_data, data_slices = flatten_structured(
            data,
            independence=independence
        )

        train_data = tree.map(
            lambda leaf: leaf[:n_simulations],
            combine_data(train_data, flat_data)
        )

        train_data = combine_data(train_data, flat_data)

        train_key, itr_key, key = jr.split(key, 3)
        train_iter, val_iter = flat_as_batch_iterators(
            itr_key,
            train_data
        )

        estim.fit(
            train_key,
            train_iter,
            val_iter,
            n_iter=n_epochs
        )


    # sample calibration data
    post_samples_cal = posterior.sample_batched((1,), x=x_cal)[0]

    flow_inverse_transform = lambda theta, x: npe.net._transform(theta, context=x)[0]
    flow_base_dist = torch.distributions.MultivariateNormal(
        torch.zeros(2), torch.eye(2)
    ) # same as npe.net._distribution

    evaluation = evaluate(
        theta_cal,
        x_cal,
        post_samples_cal,
        flow_inverse_transform,
        flow_base_dist,
        obs
    )

    return inference_results, evaluation



def fmpe_inference(
    key,
    fns,
    sample_size,
    obs,
    post_samples,
    n_iter,
    theta_cal,
    x_cal
    ):
    n_dim_theta = 26
    n_layers, hidden_size = 5, 128
    neural_network = make_cnf(
        n_dim_theta,
        n_layers,
        hidden_size
    )

    estim= SFMPE(fns, neural_network)

    key, sample_key, fit_key, post_key = jr.split(key, 4)

    data, _ = estim.simulate_data(
        sample_key,
        n_simulations=sample_size,
    )

    fmpe_params, _ = estim.fit(
        fit_key,
        data=data,
        optimizer=optax.adam(0.001),
        n_iter=n_iter,
        n_early_stopping_delta=0.00001,
        n_early_stopping_patience=30
    )

    inference_results, _ = estim.sample_posterior(
        post_key,
        fmpe_params,
        obs,
        n_samples=post_samples
    )
    # sample calibration data
    post_samples_cal = estim.sample_batched((1,), x=x_cal)[0]

    flow_inverse_transform = lambda theta, x: npe.net._transform(theta, context=x)[0]
    flow_base_dist = torch.distributions.MultivariateNormal(
        torch.zeros(2), torch.eye(2)
    ) # same as npe.net._distribution

    evaluation = evaluate(
        theta_cal,
        x_cal,
        post_samples_cal,
        flow_inverse_transform,
        flow_base_dist,
        obs
    )

    return inference_results, evaluation

def evaluate(theta, x, post, inverse, dist, obs):
    lc2st_nf = LC2ST_NF(
        thetas=theta,
        xs=x,
        posterior_samples=post,
        flow_inverse_transform=inverse,
        flow_base_dist=dist,
        num_ensemble=5,
    )
    lc2st_nf.train_under_null_hypothesis()
    lc2st_nf.train_on_observed_data()

    # Define significance level for diagnostics
    conf_alpha = 0.05

    return {
        'lc2stnf': lc2st_nf.get_statistic_on_observed_data(x_o=obs),
        'p_value': lc2st_nf.p_value(obs),
        'reject': lc2st_nf.reject_test(obs, alpha=conf_alpha)
    }

# inference function
def inference(key):
    # create ground truth
    n_sites = 20
    key, key_i = jr.split(key)
    theta_true = prior_fn().sample(seed=key_i)
    # sample v_start as integer between 0 and 100 for each site
    key, key_i = jr.split(key)

    key, sim_key = jr.split(key)
    obs = simulator_fn(sim_key, theta_true, **{
        'v_start': 0.,
        'mu': 0.,
        'obs_t': jnp.arange(100) + 1,
        'pop': 1000,
        'inf_0': 1
    })

    # sample prior
    sample_size = 10

    key, prior_key = jr.split(key)
    theta = prior_fn().sample((sample_size,), seed=prior_key)

    fns = prior_fn, simulator_fn

    n_iter = 100

    # sample calibration data
    theta_cal = prior_fn().sample((NUM_CAL,))
    x_cal = simulator_fn(theta_cal)

    # perform inference using FMPE
    post, evaluation = fmpe_inference(
        key,
        fns,
        sample_size,
        obs,
        theta,
        n_iter,
        theta_cal,
        x_cal
    )
   

    # perform inference using sFMPE
    # sfmpe_post, sfmpe_evaluation = sfmpe_inference(
        # key,
        # fns,
        # sample_size,
        # obs,
        # theta,
        # n_iter,
        # theta_cal,
        # x_cal
    # )
