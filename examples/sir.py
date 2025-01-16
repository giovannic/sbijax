# Implement an SIR model in JAX using an ODE
# and perform inference using (s)FMPE
# calculate validation statistics and save posteriors 

import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import tensorflow_probability.substrates.jax as tfp
from jax.experimental.ode import odeint

from sfmpe.fmpe import SFMPE
from sfmpe.nn.make_continuous_flow import CNF

tfd = tfp.distributions
tfb = tfp.bijectors

def prior_fn(**kwargs):
    t= kwargs['t']
    n_t= len(t)
    n_sites = kwargs['n_sites']
    v_eff = tfd.TruncatedNormal(
        loc=.8,
        scale=.1,
        low=0.,
        high=1.
    )
    r_0 = tfd.Uniform(0., 10.)

    # define a transition function which multiplies the previous
    # r number by a random draw of the F distribution
    d1, d2 = 40., 40.
    f_dist = tfb.Chain([
        tfb.Scale(d2 / d1),
        tfb.Exp()
    ])(tfd.SigmoidBeta(d1 / 2, d2 / 2))

    batch_event_shape = (t.shape[0], n_sites)
    def _transition(seed, r_t):
        return f_dist.sample( #type: ignore
            sample_shape = batch_event_shape, seed=seed
        ) * r_t[..., -1] 
        
    rt_norm = tfd.Autoregressive(
        _transition,
        sample0=jnp.ones((n_sites,)),
        num_steps=n_t
    )

    return tfd.JointDistributionNamed(dict(
        v_eff=v_eff,
        rt_norm=rt_norm,
        r_0=r_0
    ))

# interpolate function,
# return rate of infection at time t, requires t to be in (0, len(r_t)-1)
def _interpolate(t, r_t):
    return jnp.interp(t, jnp.arange(len(r_t)), r_t)

# ode function for SIRVD model including vaccination and death
def dy_dt(t, state, *args):
    s, i, _, _, _ = state
    (
        v_start,
        v_eff,
        r_t,
        gamma, #recovery
        mu #fatality
    ) = args
    if t < v_start:
        v_eff = 0.

    beta = _interpolate(t, r_t) * gamma
    return jnp.array([
        -beta * s * i - v_eff * s, # susceptible
        beta * s * i - gamma * i - mu * i, # infected
        gamma * i, # recovered
        v_eff * s, # vaccinated
        mu * i # deceased
    ])

def simulator_fn(seed, theta, **kwargs):
    v_start = kwargs['v_start']
    v_eff = theta['v_eff']
    r_t = theta['rt_norm'] * theta['r_0']
    gamma = theta['gamma']
    mu = kwargs['mu']
    obs_t = kwargs['obs_t']
    pop = kwargs['pop']
    inf_0 = kwargs['inf_0']

    # add t = 0 to obs_t
    obs_t = jnp.concatenate([
        jnp.array([jnp.newaxis, jnp.newaxis, 0.]),
        obs_t
    ])

    # solve ode
    y0 = jnp.array([pop - inf_0, inf_0, 0., 0., 0.])
    y = vmap(
        odeint,
        in_axes=(
            None, #dy_dt
            None, #y0
            0, #obs_t
            0, #v_start
            None, #v_eff
            0, #r_t
            None, #gamma
            None #mu
        )
    )(
        dy_dt,
        y0,
        obs_t,
        v_start,
        v_eff,
        r_t,
        gamma,
        mu
    )
    deaths = y[:, 4]
    new_deaths = jnp.diff(deaths)

    # simulate observed deaths according to a poisson distribution
    return {
        'obs_deaths': tfd.Poisson(new_deaths).sample(seed=seed),
        'v_start': v_start
    }

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



    return inference_results, estim, fmpe_params

def sfmpe_inference(
    key,
    fns,
    sample_size,
    obs,
    post_samples,
    n_iter,
    theta_cal,
    x_cal
    ):
    key, sample_key, fit_key, post_key = jr.split(key, 4)
    rngs = nnx.Rngs(fit_key)
    config = {
        'latent_dim': 12,
        'label_dim': 2,
        'index_out_dim': 4,
        'n_encoder': 2,
        'n_decoder': 2,
        'n_heads': 2,
        'n_ff': 2,
        'dropout': .1,
        'activation': nnx.relu,
    }
    nn = Transformer(
        config,
        n_context_labels=1,
        context_index_dim=1,
        n_theta_labels=2,
        theta_index_dim=1,
        rngs=rngs
    )
    theta_index_size= 3
    model = CNF(theta_dim=theta_index_size*2, transform=nn)

    estim = SFMPE(
        fns,
        model
    )

    data = estim.simulate_data(
        sample_key,
        n_simulations=sample_size
    )

    estim.fit(
        fit_key,
        data=data,
        optimizer=optax.adam(0.001),
        n_iter=n_iter,
        n_early_stopping_delta=0.00001,
        n_early_stopping_patience=30
    )

    inference_results, _ = estim.sample_posterior(
        post_key,
        obs,
        n_samples=post_samples
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

    # write posteriors to file
    post.to_netcdf(outdir + 'post.netcdf')

    # write evaluation dictionary to file
    with open(outdir + 'evaluation.json', 'w') as f:
        json.dump(evaluation, f)
