import wandb
wandb.login()

sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'test_bpd',
        'goal': 'minimize'
    },
    'parameters': {
        # Data
        'dataset': {'values': ["NATURE", "CHEETAH", "PEIP"]},
        'size': {'values': [5, 10]},
        'epochs': {'values': [5, 200]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="Normalizing_Flows")
print("Strated sweep!")
print("Sweep ID:", sweep_id)
