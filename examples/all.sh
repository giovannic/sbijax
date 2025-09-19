
export JAX_ENABLE_X64=1
python examples/hierarchical_gaussian.py -m n_simulations=1000,5000,10000 n_theta=50 n_rounds=1 n_epochs=1000 seed=0,1,2,3,4
python examples/hierarchical_gaussian.py -m n_simulations=5000 n_rounds=1 n_epochs=1000 n_theta=1,2,5,10,20,50 seed=0,1,2,3,4
python examples/hierarchical_brownian.py -m n_simulations=1000,5000,10000 n_theta=50 n_rounds=1 n_epochs=1000 seed=0,1,2,3,4 f_in_sample=prior
python examples/hierarchical_brownian.py -m n_simulations=5000 n_rounds=1 n_epochs=1000 n_theta=1,2,5,10,20,50,100 seed=0,1,2,3,4 f_in_sample=prior
python examples/seir.py -m n_simulations=1000,5000,10000 n_sites=50 n_obs=5 n_rounds=1 n_epochs=1000 seed=0,1,2,3,4 f_in_sample=prior 'inference.sample_param
s=["beta_0","A"]'
python examples/seir.py -m n_simulations=5000 n_rounds=1 n_epochs=1000 n_obs=5 n_sites=1,2,5,10,20,50 seed=0,1,2,3,4 f_in_sample=prior 'inference.sample_param
s=["beta_0","A"]'
